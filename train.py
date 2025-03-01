from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, ClassLabel
import torch

dataset = load_dataset("csv", data_files="training_data.csv")

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

class_labels = ClassLabel(names=["doomer", "not doomer", "neutral"])
split_dataset = split_dataset.cast_column("sentiment", class_labels)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    result = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=128)
    result["labels"] = examples["sentiment"]
    return result

tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

tokenized_columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset["train"].column_names if col not in tokenized_columns]
)

tokenized_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label={i: label for i, label in enumerate(class_labels.names)}
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

model.save_pretrained("./doomer_detector")
tokenizer.save_pretrained("./doomer_detector")

results = trainer.evaluate()
print(f"Evaluation results:", results)