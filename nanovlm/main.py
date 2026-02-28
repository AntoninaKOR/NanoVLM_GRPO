import torch
from typing import Dict, List, Optional

def train(
    model: None,
    trajectories: List[List[Dict]],
    mode: str = "action",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-5,
    device: Optional[torch.device] = None,
):
    assert model is not None, "Model must be provided for training"
    assert mode in {"action", "reward"}, "Mode must be either 'action' or 'reward'"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    examples = prepare_training_data(trajectories, mode=mode)
    dataset = TrajectoryDataset(examples, prompt=model.prepare_prompt(mode))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, labels = _collate_batch(
                batch,
                tokenizer=model.tokenizer,
                image_processor=model.image_processor,
                device=device,
            )
            outputs = model.model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

def _collate_batch(
    batch: List[Dict[str, any]],
    tokenizer,
    image_processor,
    device: torch.device,
):
    prompts = [item["prompt"] for item in batch]
    targets = [item["target"] for item in batch]
    images = [item["image"] for item in batch]

    # Build full text for causal LM training
    full_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]

    text_inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Mask prompt tokens so loss focuses on target
    prompt_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    labels = text_inputs["input_ids"].clone()
    for i in range(labels.size(0)):
        prompt_len = int(prompt_tokens["input_ids"][i].ne(tokenizer.pad_token_id).sum())
        labels[i, :prompt_len] = -100

    inputs = dict(text_inputs)
    if image_processor is not None:
        image_inputs = image_processor(images=images, return_tensors="pt")
        inputs.update(image_inputs)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    return inputs, labels


if __name__ == "__main__":
    print("Training loop is not implemented yet. Please implement the model and provide trajectories for training.")    