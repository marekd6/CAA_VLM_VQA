import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from os.path import join
from tqdm import tqdm


behaviour = 'count'
beh_answers = [str(i) for i in range(10)] if behaviour == 'count' else []

model_id = "google/gemma-3-4b-it"

ds_dir_fn='/content/drive/MyDrive/clevr/caa_cnt_val'
vect_dir_fn = '/content/drive/MyDrive/clevr/caa_cnt_train'
img_pth = '/content/CLEVR_v1.0/images/val'
layers = [15, 14, 13]
idx = 0  # global sample index


def load_model():
  processor = AutoProcessor.from_pretrained(model_id)
  model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16
  ).eval()
  return processor, model


class CLEVRExtractionDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} extraction pairs.")
        self.image_dir = Path(image_dir)
        if len(self.data) == 0:
            raise ValueError("No data")

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        q = self.data[idx]
        image = Image.open(join(self.image_dir, q['image_path'])).convert('RGB')

        return {
            "image": image,
            "question": q['question'],
            "true_answer": str(q['true_answer']),
            "target_answer": str(q['target_answer'])
        }


def pil_collate_fn(batch):
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def make_hook(layer, buf, key):
    def hook(module, input, output):
        buf[layer][idx:idx+output.size(0)] = output[:, -1, :].detach()
    return hook


def generate_save_vectors_for_behavior(model, processor, dataloader, fn, target_layers=[15]):
    model.eval()
    idx = 0  # global sample index
    N = len(dataloader.dataset)
    D = model.config.hidden_size

    pos_buf = {layer: torch.zeros((N, D), device=model.device) for layer in target_layers}
    neg_buf = {layer: torch.zeros((N, D), device=model.device) for layer in target_layers}

    print(f"Extracting activations from layers: {target_layers}")

    hooks = []
    for layer in target_layers:
        module = dict(model.named_modules())[layer]
        hooks.append(module.register_forward_hook(make_hook(layer, pos_buf, "pos")))
        hooks.append(module.register_forward_hook(make_hook(layer, neg_buf, "neg")))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Extraction Pairs"):
            bsz = batch["image"].size(0)
            q = batch["question"][0]
            true_ans = batch["true_answer"][0]
            target_ans = batch["target_answer"][0]
            image = batch["image"][0]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }
            ]
            base_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            pos_text = base_prompt + target_ans
            neg_text = base_prompt + true_ans
            pos_inputs = processor(text=pos_text, images=image, return_tensors="pt").to(model.device)
            neg_inputs = processor(text=neg_text, images=image, return_tensors="pt").to(model.device)

            model(**pos_inputs, output_hidden_states=True)
            model(**neg_inputs, output_hidden_states=True)

            idx += bsz

    for h in hooks:
        h.remove()

    for layer in target_layers:
        steering = (pos_buf[layer] - neg_buf[layer]).mean(dim=0)
        torch.save(steering.cpu(), f"{fn}_{layer}.pt")
        print("saved", fn, layer)


def gen_ds_steering_vect(mod, proc, ds='100', layers=layers, ds_dir_fn=ds_dir_fn, vect_dir_fn=vect_dir_fn):
    extract_dataset = CLEVRExtractionDataset(f'{ds_dir_fn}_{ds}.jsonl', img_pth)
    extract_loader = DataLoader(extract_dataset, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)
    generate_save_vectors_for_behavior(
        model=mod,
        processor=proc, 
        dataloader=extract_loader, 
        fn=f"{vect_dir_fn}_{ds}", 
        target_layers=layers
    )


def main():
    proc, mod = load_model()
    gen_ds_steering_vect(mod, proc, '10')
    gen_ds_steering_vect(mod, proc, '100')
    gen_ds_steering_vect(mod, proc, '1000')


if __name__ == '_main_':
    main()
