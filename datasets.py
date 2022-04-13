import numpy as np
from transformers import CLIPProcessor
import torch
from PIL import Image
from torch.utils.data import Dataset

def context_padding(inputs, context_length=77):
    shape = (1,context_length - inputs.input_ids.shape[1])
    x = torch.zeros(shape)
    input_ids = torch.cat([inputs.input_ids,x], dim=1).long()
    attention_mask = torch.cat([inputs.attention_mask,x], dim=1).long()
    return input_ids, attention_mask

class Dataset(Dataset):
    def __init__(self, df, images_path):
        super().__init__()
        self.df = df
        self.images_path = images_path
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.images_path}/{row['link'][-19:]}.jpg")
        inputs = self.processor(text = row['title'], images = image, return_tensors="pt", padding=True)
        input_ids, attention_mask = context_padding(inputs)
        label = torch.from_numpy(np.asarray(row['label']))
        return input_ids.squeeze(), attention_mask.squeeze(), inputs.pixel_values.squeeze(), label