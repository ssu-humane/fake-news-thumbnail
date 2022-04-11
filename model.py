from transformers import CLIPModel
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32"):
        super(ClassificationModel, self).__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_model)
        self.bilayer = nn.Bilinear(512, 512, 512)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(512, 1)
                
    def forward(self, input_ids, attention_mask, pixel_values):
        clip_layer  = self.clip(input_ids= input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        x = self.bilayer(clip_layer.text_embeds, clip_layer.image_embeds)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        return self.linear2(x)
    
    def clip_freeze(self):
        model_weight = self.clip.state_dict().keys()
        model_weight_list = [*model_weight]
        for name, param in self.clip.named_parameters():
            if name in model_weight_list:
                param.requires_grad = False
                