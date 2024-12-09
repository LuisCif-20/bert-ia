from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label)
        return item

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def train_model(train_texts, train_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=7)  # Asegúrate de que num_labels=7

    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    training_args = TrainingArguments(
        output_dir='./models/chatbot_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=None,  # Use default data collator
    )

    trainer.train()
    model.save_pretrained('./models/chatbot_model')
    tokenizer.save_pretrained('./models/chatbot_model')

if __name__ == "__main__":
    texts, labels = load_data('data/dialogues_processed.csv')
    train_model(texts, labels)
