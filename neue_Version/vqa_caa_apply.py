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

HF = ''


behaviour = 'count'
opt = '_opt'
opt = ''

model_id = "google/gemma-3-4b-it"

data_dir = ''
save_dir = ''

ds_dir_fn = f'{data_dir}/caa_{behaviour}_val{opt}'
vect_dir_fn = f'{data_dir}/caa_{behaviour}_train{opt}'
img_pth = f'{save_dir}/images/val'
layers = [x for x in range(34)]
multipliers = [x / 2.0 for x in range(-4, 4, 1)] + [2.0]


def gen_true_answers() -> list:
    if behaviour == 'count':
        return [str(i) for i in range(11)]
    return ['2']

beh_answers = gen_true_answers()


def gen_answ_options() -> str:
    suffix = '\nChoices:'
    for i, answ in enumerate(beh_answers):
        suffix += f'\n{i}{answ}'
    return suffix


beh_answers = beh_answers if opt == '' else gen_answ_options()


def load_model():
  processor = AutoProcessor.from_pretrained(model_id)
  model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    token=HF
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
            answ_token_probabs = F.softmax(answ_token_logits.to(torch.float32), dim=-1).squeeze()
            prob_dict = {answ: answ_token_probabs[token].item() for answ, token in token_map.items()}

            stats = metrics[q_type]
            stats["total"] += 1
            stats["sum_prob_true"] += prob_dict.get(true_ans, 0.0)
            stats["sum_prob_target"] += prob_dict.get(target_ans, 0.0)

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
    extract_dataset = CLEVRCAAEvaluationDataset(f'{ds_dir_fn}_{ds}.jsonl', img_pth)
    dl = DataLoader(extract_dataset, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

    for target_layer in reversed(layers):
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
    gc.collect()
    torch.cuda.empty_cache()
    try:
      proc, mod = load_model()
      answ_tokens = valid_answ_tokens2(beh_answers, proc)
      eval_ds_steered(mod, proc, answ_tokens, '10')
      eval_ds_steered(mod, proc, answ_tokens, '100')
      eval_ds_steered(mod, proc, answ_tokens, '500')
      eval_ds_steered(mod, proc, answ_tokens, '1000')
    finally:
      del proc
      del mod
      del answ_tokens
      gc.collect()
      torch.cuda.empty_cache()

if __name__ == '_main_':
    main()
