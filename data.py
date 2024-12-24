from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def download_and_process_data():
    dataset = load_dataset("ybelkada/football-dataset", split="train")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return dataset, processor
