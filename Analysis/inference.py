import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_model():
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    categories = ["Ethnicity/Race", "Not Cyberbullying", "Gender/Sexual", "Religion"]
    return categories[prediction]
