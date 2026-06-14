import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


behaviour = 'count_2_4'

model_id = "google/gemma-3-4b-it"

ds_dir_fn = f'/content/drive/MyDrive/clevr/caa_{behaviour}_train'
vect_dir_fn = ds_dir_fn
img_pth = '/content/CLEVR_v1.0/images/val'
layers = [x for x in range(34)]
multipliers = [x / 2.0 for x in range(-4, 4, 1)] + [2.0]


def gen_true_answ():
    if behaviour == 'count':
        return [str(i) for i in range(10)]
    if behaviour == 'count_2_4':
        return ['2']
    return ['2']


beh_answers = gen_true_answ()


def behaviour_cond(true_answer):
    return str(true_answer) in beh_answers


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
        q_type = behaviour if behaviour_cond(true_answer) else "other"

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
    return {answ: proc.tokenizer.encode(answ, add_special_tokens=False) for answ in valid_answers}


class SteeringHook:
    def __init__(self, model, steering_vector, target_layer, coefficient):
        self.model = model
        self.steering_vector = steering_vector
        self.target_layer = target_layer
        self.coefficient = coefficient
        self.handle = None
        # self.text_layers = model.model.language_model.layers   # 34 Gemma3DecoderLayer blocks

        # hooks = []

        # for idx in target_layers:
        #     layer_module = text_layers[idx]
        #     hooks.append(layer_module.register_forward_hook(make_hook(idx, pos_buf, neg_buf)))

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
        "sum_prob_target": 0.0
    })

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval multiplier: {multiplier}"):
            q = batch["question"][0]
            image = batch["image"][0]
            true_ans = batch["true_answer"][0]
            target_ans = batch["target_answer"][0]
            q_type = batch["q_type"][0]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }
            ]

            eval_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(text=eval_prompt, images=image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            
            answ_token_logits = outputs.logits[:, -1, :]
            answ_token_probabs = F.softmax(answ_token_logits.to(torch.float32), dim=-1)
            prob_dict = {answ: answ_token_probabs[token].item() for answ, token in token_map}

            stats = metrics[q_type]
            stats["total"] += 1
            stats["p_true"] += prob_dict.get(true_ans, 0.0)
            stats["p_target"] += prob_dict.get(target_ans, 0.0)

    hook.remove()
    
    final_metrics = {}
    for q_type, stats in metrics.items():
        total = stats["total"]
        final_metrics[q_type] = {
            "avg_prob_true": stats["sum_prob_true"] / total if total > 0 else 0,
            "avg_prob_target": stats["sum_prob_target"] / total if total > 0 else 0
        }
        
    return final_metrics


def eval_ds_steered(mod, proc, answ_tokens, ds='100', layers=layers, ds_dir_fn=ds_dir_fn, vect_dir_fn=vect_dir_fn, multipliers=multipliers):
    extract_dataset = CLEVRCAAEvaluationDataset(f'{ds_dir_fn}{ds}.jsonl', img_pth)
    dl = DataLoader(extract_dataset, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

    for target_layer in layers:
        print("Evaluating on layer", target_layer)
        steering_vector = torch.load(f"{vect_dir_fn}_{ds}_{target_layer}.pt")

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
                avg_true = stats["avg_prob_true"]
                avg_target = stats["avg_prob_target"]
                print(mult, avg_true, avg_target)

        baseline_metrics = sweep_results[0.0]

        for mult, metrics in sweep_results.items():
            print("Results with", mult, 'multiplier')
            for q_type, stats in metrics.items():
                avg_true = stats["avg_prob_true"]
                avg_target = stats["avg_prob_target"]
        
                shift_true = avg_true - baseline_metrics[q_type]["avg_prob_true"]
                shift_target = avg_target - baseline_metrics[q_type]["avg_prob_target"]

                print(mult, q_type, 'P(True)', avg_true, 'd(True)', shift_true, 'P(Target)', avg_target, 'd(Target)', shift_target)
        
        with open(f"{vect_dir_fn}_{ds}_{target_layer}.json", "w") as f:
            json.dump(sweep_results, f, indent=4)


def main():
    proc, mod = load_model()
    answ_tokens = valid_answ_tokens2(beh_answers, proc)
    eval_ds_steered(mod, proc, answ_tokens, '10')
    eval_ds_steered(mod, proc, answ_tokens, '100')
    eval_ds_steered(mod, proc, answ_tokens, '1000')


if __name__ == '_main_':
    main()
