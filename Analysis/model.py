import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

filepath = "C:/Users/chiki/OneDrive/Desktop/p4,5/sem5/DAV/assignment/preprocessed_data.csv"

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data['cleaned_text']
    y = data['label_encoded']
    return X, y

# Dataset preparation
class CyberbullyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(filepath):
    # Load data
    X, y = load_data(filepath)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

    # Prepare datasets
    train_dataset = CyberbullyDataset(X_train, y_train, tokenizer)
    val_dataset = CyberbullyDataset(X_val, y_val, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Reduced batch size
        gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
        evaluation_strategy="epoch",
        save_steps=500,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=True  # Enable mixed precision training
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train and save
    trainer.train()
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model training complete!")

# Train the model
train_model(filepath)
