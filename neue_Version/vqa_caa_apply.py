import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import gc
from vqa_caa_create_double import SYS_PROMP


behaviour = 'count'
opt = '_opt'
# opt = ''

model_id = "google/gemma-3-4b-it"

data_dir = '/data/5drwal'
clevr_dir = '/informatik/wtm/datasets/External Datasets/CLEVR/CLEVR_v1.0'

ds_dir_fn = f'{data_dir}/caa_{behaviour}_val{opt}'
vect_dir_fn = f'{data_dir}/caa_{behaviour}_train{opt}'
vect_dir_fn = f'{data_dir}/caa_{behaviour}_train{opt}' + '_SYMPR'
res_sub_dir = f'{data_dir}/res/caa_{behaviour}_train{opt}'
res_sub_dir = f'{data_dir}/res_SYMPR/caa_{behaviour}_train{opt}'
img_pth = f'{clevr_dir}/images/val'
layers = [x for x in range(34)]
multipliers = [x / 2.0 for x in range(-4, 4, 1)] + [2.0]


def gen_true_answers() -> list:
    if behaviour == 'count':
        return [str(i) for i in range(11)]
    return ['2']

beh_answers = gen_true_answers()
beh_answers = beh_answers if opt == '' else [str(i) for i in range(len(beh_answers))]


def load_model():
  processor = AutoProcessor.from_pretrained(model_id)
  model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16
  ).eval()
  return processor, model


class CLEVRCAAEvaluationDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} qs.")
        self.image_dir = Path(image_dir)
        if len(self.data) == 0:
            raise ValueError("No data")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        q = self.data[idx]
        image_path = self.image_dir / q['image_path']
        image = Image.open(image_path).convert('RGB')
        true_answer = str(q['true_answer'])
        q_type = q['q_type']

        return {
            "image": image,
            "question": q['question'],
            "true_answer": true_answer,
            "target_answer": str(q['target_answer']),
            "q_type": q_type
        }


def pil_collate_fn(batch):
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def valid_answ_tokens2(valid_answers, proc):
    return {answ: proc.tokenizer.encode(answ, add_special_tokens=False)[0] for answ in valid_answers}


class SteeringHook:
    def __init__(self, model, steering_vector, target_layer, coefficient):
        self.model = model
        self.steering_vector = steering_vector
        self.target_layer = target_layer
        self.coefficient = coefficient
        self.handle = None

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output

        sv = self.steering_vector.to(activations.device, dtype=activations.dtype)

        if sv.dim() == 1:
            sv = sv.view(1, 1, -1)

        activations = activations + self.coefficient * sv

        if isinstance(output, tuple):
            return (activations,) + output[1:]

        return activations

    def attach(self):
        layer_module = self.model.model.language_model.layers[self.target_layer]
        self.handle = layer_module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def evaluate_steering_vector(model, processor, dataloader, steering_vector, target_layer, multiplier, token_map):
    hook = SteeringHook(model, steering_vector, target_layer, multiplier)
    hook.attach()
    model.eval()

    metrics = defaultdict(lambda: {
        "total": 0,
        "sum_prob_true": 0.0,
        "sum_prob_target": 0.0,
        "sum_prob_true_norm": 0.0,
        "sum_prob_target_norm": 0.0,
        "sum_prob_true_norm2": 0.0,
        "sum_prob_target_norm2": 0.0,
        "sum_prob_true_filtered": 0.0,
        "sum_prob_target_filtered": 0.0,
    })

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval multiplier: {multiplier}"):
            q = batch["question"][0]
            image = batch["image"][0]
            true_ans = batch["true_answer"][0]
            target_ans = batch["target_answer"][0]
            q_type = batch["q_type"][0]

            messages = [{
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYS_PROMP}
                    ]
                }] if len(SYS_PROMP) > 0 else []
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }
            )

            eval_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(text=eval_prompt, images=image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)

            answ_token_logits = outputs.logits[:, -1, :].to(torch.float32)
            answ_token_probabs = F.softmax(answ_token_logits, dim=-1).squeeze()
            answ_token_probabs_norm = F.softmax(answ_token_logits[:, list(token_map.values())], dim=-1).squeeze()

            prob_dict = {answ: answ_token_probabs[token].item() for answ, token in token_map.items()}
            prob_dict_norm = {answ: answ_token_probabs_norm[i].item() for i, answ in enumerate(token_map.keys())}
            norm_prob = get_valid_probs(token_map, prob_dict)
            norm_prob2 = prob_dict_norm # get_valid_probs2(token_map, prob_dict2)
            filtered_prob_dict = top_p_probs(token_map, answ_token_probabs)

            top_prob, top_token_id = torch.max(answ_token_probabs, dim=-1)
            top_token_text = processor.tokenizer.decode(top_token_id)
            print(f"Top overall prediction: '{top_token_text}' with probability: {top_prob.item():.4f}")
            best_valid_answer = max(prob_dict, key=prob_dict.get)
            best_valid_prob = prob_dict[best_valid_answer]
            print(f"Top valid answer: '{best_valid_answer}' (Share of Top-P: {best_valid_prob})")

            stats = metrics[q_type]
            stats["total"] += 1
            stats["sum_prob_true"] += prob_dict.get(true_ans, 0.0)
            stats["sum_prob_target"] += prob_dict.get(target_ans, 0.0)

            stats["sum_prob_true_norm"] += norm_prob.get(true_ans, 0.0)
            stats["sum_prob_target_norm"] += norm_prob.get(target_ans, 0.0)
            stats["sum_prob_true_norm2"] += norm_prob2.get(true_ans, 0.0)
            stats["sum_prob_target_norm2"] += norm_prob2.get(target_ans, 0.0)

            stats["sum_prob_true_filtered"] += filtered_prob_dict.get(true_ans, 0.0)
            stats["sum_prob_target_filtered"] += filtered_prob_dict.get(target_ans, 0.0)


    hook.remove()

    final_metrics = {}
    for q_type, stats in metrics.items():
        total = stats["total"]

        if total > 0:
            final_metrics[q_type] = {
                "avg_prob_true":   stats["sum_prob_true"]   / total,
                "avg_prob_target": stats["sum_prob_target"] / total,

                "avg_prob_true_norm":   stats["sum_prob_true_norm"]   / total,
                "avg_prob_target_norm": stats["sum_prob_target_norm"] / total,

                "avg_prob_true_norm2":   stats["sum_prob_true_norm2"]   / total,
                "avg_prob_target_norm2": stats["sum_prob_target_norm2"] / total,

                "avg_prob_true_filtered":   stats["sum_prob_true_filtered"]   / total,
                "avg_prob_target_filtered": stats["sum_prob_target_filtered"] / total,
            }
        else:
            final_metrics[q_type] = {
                "avg_prob_true": 0.0,
                "avg_prob_target": 0.0,
                "avg_prob_true_norm": 0.0,
                "avg_prob_target_norm": 0.0,
                "avg_prob_true_norm2": 0.0,
                "avg_prob_target_norm2": 0.0,
                "avg_prob_true_filtered": 0.0,
                "avg_prob_target_filtered": 0.0,
            }


    return final_metrics


def get_valid_probs(token_map, prob_dict):
    valid_probs = torch.tensor([prob_dict[a] for a in token_map.keys()])
    valid_sum = valid_probs.sum().item()

    if valid_sum > 0:
        norm_prob = {a: prob_dict[a] / valid_sum for a in token_map.keys()}
    else:
        norm_prob = {a: 0.0 for a in token_map.keys()}
    return norm_prob


def top_p_probs(token_map, full_vocab_probs, threshold=0.7):
    # 1. Sort the ENTIRE vocabulary probabilities
    sorted_probs, sorted_vocab_ids = torch.sort(full_vocab_probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # 2. Create the mask for Top-P
    mask = cumulative <= threshold
    mask[0] = True  # Safety: always keep the top token
    
    # 3. Extract just the Top-P subset
    top_p_probs = sorted_probs[mask]
    top_p_vocab_ids = sorted_vocab_ids[mask]
    
    # Calculate the sum of the top-p subset for normalization
    top_p_sum = top_p_probs.sum().item()

    # Create a dictionary for instantly looking up if a token made the cut: {vocab_id: probability}
    top_p_dict = {
        vid.item(): p.item() 
        for vid, p in zip(top_p_vocab_ids, top_p_probs)
    }

    # 4. Intersect with your valid answers
    filtered_prob_dict = {}
    for answ, token_id in token_map.items():
        if token_id in top_p_dict:
            # The valid answer survived the top-p filter! 
            # Normalize its probability against the top-p subset's total mass.
            filtered_prob_dict[answ] = top_p_dict[token_id] / top_p_sum

    return filtered_prob_dict


def eval_ds_steered(mod, proc, answ_tokens, ds='100', vec_ds='100', layers=layers, ds_dir_fn=ds_dir_fn, vect_dir_fn=vect_dir_fn, multipliers=multipliers):
    extract_dataset = CLEVRCAAEvaluationDataset(f'{ds_dir_fn}_{ds}.jsonl', img_pth)
    dl = DataLoader(extract_dataset, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

    for target_layer in reversed(layers):
        print("Evaluating on layer", target_layer, 'for ds', ds, 'with vec size', vec_ds)
        steering_vector = torch.load(f"{vect_dir_fn}_{vec_ds}_{target_layer}.pt")

        sweep_results = {}
        for mult in multipliers:
            print("Evaluating with", mult, 'multiplier')
            sweep_results[mult] = evaluate_steering_vector(
                model=mod,
                processor=proc,
                dataloader=dl,
                steering_vector=steering_vector,
                target_layer=target_layer,
                multiplier=mult,
                token_map=answ_tokens
            )
            for q_type, stats in sweep_results[mult].items():
                print(
                    mult,
                    stats["avg_prob_true"],
                    stats["avg_prob_target"],
                    stats["avg_prob_true_norm"],
                    stats["avg_prob_target_norm"],
                    stats["avg_prob_true_norm2"],
                    stats["avg_prob_target_norm2"],
                    stats["avg_prob_true_filtered"],
                    stats["avg_prob_target_filtered"],
                )

        baseline_metrics = sweep_results[0.0]

        for mult, metrics in sweep_results.items():
            print("Results with", mult, "multiplier")

            for q_type, stats in metrics.items():
                base = baseline_metrics[q_type]

                # --- RAW ---
                shift_true = stats["avg_prob_true"] - base["avg_prob_true"]
                shift_target = stats["avg_prob_target"] - base["avg_prob_target"]

                rel_true = shift_true / base["avg_prob_true"] if base["avg_prob_true"] != 0 else 0.0
                rel_target = shift_target / base["avg_prob_target"] if base["avg_prob_target"] != 0 else 0.0

                # --- NORM ---
                shift_true_norm = stats["avg_prob_true_norm"] - base["avg_prob_true_norm"]
                shift_target_norm = stats["avg_prob_target_norm"] - base["avg_prob_target_norm"]

                rel_true_norm = shift_true_norm / base["avg_prob_true_norm"] if base["avg_prob_true_norm"] != 0 else 0.0
                rel_target_norm = shift_target_norm / base["avg_prob_target_norm"] if base["avg_prob_target_norm"] != 0 else 0.0

                # --- NORM2 ---
                shift_true_norm2 = stats["avg_prob_true_norm2"] - base["avg_prob_true_norm2"]
                shift_target_norm2 = stats["avg_prob_target_norm2"] - base["avg_prob_target_norm2"]

                rel_true_norm2 = shift_true_norm2 / base["avg_prob_true_norm2"] if base["avg_prob_true_norm2"] != 0 else 0.0
                rel_target_norm2 = shift_target_norm2 / base["avg_prob_target_norm2"] if base["avg_prob_target_norm2"] != 0 else 0.0

                # --- FILTERED ---
                shift_true_filtered = stats["avg_prob_true_filtered"] - base["avg_prob_true_filtered"]
                shift_target_filtered = stats["avg_prob_target_filtered"] - base["avg_prob_target_filtered"]

                rel_true_filtered = shift_true_filtered / base["avg_prob_true_filtered"] if base["avg_prob_true_filtered"] != 0 else 0.0
                rel_target_filtered = shift_target_filtered / base["avg_prob_target_filtered"] if base["avg_prob_target_filtered"] != 0 else 0.0

                # --- SAVE EVERYTHING ---
                stats.update({
                    "shift_true": shift_true,
                    "shift_target": shift_target,
                    "shift_true_norm": shift_true_norm,
                    "shift_target_norm": shift_target_norm,
                    "shift_true_norm2": shift_true_norm2,
                    "shift_target_norm2": shift_target_norm2,
                    "shift_true_filtered": shift_true_filtered,
                    "shift_target_filtered": shift_target_filtered,

                    "rel_true": rel_true,
                    "rel_target": rel_target,
                    "rel_true_norm": rel_true_norm,
                    "rel_target_norm": rel_target_norm,
                    "rel_true_norm2": rel_true_norm2,
                    "rel_target_norm2": rel_target_norm2,
                    "rel_true_filtered": rel_true_filtered,
                    "rel_target_filtered": rel_target_filtered,
                })

                print(
                    mult, q_type,
                    "RAW:", stats["avg_prob_true"], shift_true, rel_true,
                    stats["avg_prob_target"], shift_target, rel_target,
                    "NORM:", stats["avg_prob_true_norm"], shift_true_norm, rel_true_norm,
                    stats["avg_prob_target_norm"], shift_target_norm, rel_target_norm,
                    "NORM2:", stats["avg_prob_true_norm2"], shift_true_norm2, rel_true_norm2,
                    stats["avg_prob_target_norm2"], shift_target_norm2, rel_target_norm2,
                    "FILT:", stats["avg_prob_true_filtered"], shift_true_filtered, rel_true_filtered,
                    stats["avg_prob_target_filtered"], shift_target_filtered, rel_target_filtered
                )


        with open(f"{res_sub_dir}_{ds}_{vec_ds}_{target_layer}.json", "w") as f:
            json.dump(sweep_results, f, indent=4)



def main():
    print('a')
    gc.collect()
    torch.cuda.empty_cache()
    try:
      print('b')
      proc, mod = load_model()
      answ_tokens = valid_answ_tokens2(beh_answers, proc)
      print('token mapping')
      for a, t in answ_tokens.items():
          print(a, t)
      eval_ds_steered(mod, proc, answ_tokens, '10', '10')
      eval_ds_steered(mod, proc, answ_tokens, '10', '100')
      eval_ds_steered(mod, proc, answ_tokens, '100', '100')
    #   eval_ds_steered(mod, proc, answ_tokens, '500')
      print('c')
    finally:
      del proc
      del mod
      del answ_tokens
      gc.collect()
      torch.cuda.empty_cache()
      print('fin')

if __name__ == '__main__':
    main()
