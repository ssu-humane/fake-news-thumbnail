import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from ..clip.modeling_clip import CLIPOutput
from transformers import LongformerModel, CLIPModel
from ...file_utils import ModelOutput
from ...modeling_outputs import BaseModelOutputWithPooling


class BTOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_title: torch.FloatTensor = None
    logits_per_body: torch.FloatTensor = None
    body_embeds: torch.FloatTensor = None
    title_embeds: torch.FloatTensor = None
    body_model_output: BaseModelOutputWithPooling = None
    title_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["body_model_output", "title_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class BIOutputs(ModelOutput):  # 바꾸기
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_body: torch.FloatTensor = None
    body_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    body_model_output: BaseModelOutputWithPooling = None
    image_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["body_model_output", "image_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TIOutputs(ModelOutput):  # 바꾸기
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_title: torch.FloatTensor = None
    title_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    title_model_output: BaseModelOutputWithPooling = None
    image_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["title_model_output", "image_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()

        )

class LongClipOutput(ModelOutput):
    BIoutputs: BaseModelOutputWithPooling = None
    BToutputs: BaseModelOutputWithPooling = None
    TIoutputs: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:  # one by one  change
        return tuple(getattr(self, k) for k in self.keys())


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class LongClipModel(nn.Module):
    def __init__(self):
        super(LongClipModel, self).__init__()
        self.bodyModel = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.clipModel = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.body_projection = nn.Linear(768,512, bias=False)
        self.loss = None

    def all_freeze(self):
        clip_weight = self.clipModel.state_dict().keys()
        clip_weight_list = [*clip_weight]
        for name, param in self.clipModel.named_parameters():
            if name in clip_weight_list:
                param.requires_grad = False
    def title_freeze(self):
        clip_weight = self.clipModel.text_Model.state_dict().keys()
        clip_weight_list = [*clip_weight]
        for name, param in self.clipModel.text_Model.named_parameters():
            if name in clip_weight_list:
                param.requires_grad = False
    def image_freeze(self):
        clip_weight = self.clipModel.vision_Model.state_dict().keys()
        clip_weight_list = [*clip_weight]
        for name, param in self.clipModel.vision_Model.named_parameters():
            if name in clip_weight_list:
                param.requires_grad = False

    def multiple_BIoutput(self, body_outputs, image_outputs):

        body_embeds = body_outputs[1]
        print(body_embeds.shape) #50, 768
        body_embeds = self.body_projection(body_embeds)
        #but text projection -> nn.linear(512, 512)'
        #self.clipModel.text_embed_dim = body_embeds.shape[1]
        #print(self.clipModel.text_embed_dim )
        #body_embeds = self.clipModel.text_projection(body_embeds)

        image_embeds = image_outputs[1]
        
        #image_embeds = self.clipModel.visual_projection(image_embeds)

        body_embeds = body_embeds / body_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.clipModel.logit_scale.exp()
        print("body: ",body_embeds.shape)
        print("image: ",image_embeds.shape)

        logits_per_body = torch.matmul(body_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_body.T

        loss = clip_loss(logits_per_body)

        return BIOutputs(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_body=logits_per_body,
            body_embeds=body_embeds,
            image_embeds=image_embeds,
            body_model_output=body_outputs,
            image_model_output=image_outputs
        )

    def multiple_BToutput(self, body_outputs, title_outputs):
        body_embeds = body_outputs[1]
        self.clipModel.text_embed_dim = body_embeds.shape[1]
        #body_embeds = self.clipModel.visual_projection(body_embeds) #50,768  // 안맞음512,512
        #body_embeds = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        body_embeds = self.body_projection(body_embeds)
        
        title_embeds = title_outputs[1] #50,512
        title_embeds = self.clipModel.text_projection(title_embeds) 

        body_embeds = body_embeds / body_embeds.norm(dim=-1, keepdim=True)
        title_embeds = title_embeds / title_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.clipModel.logit_scale.exp()
        logits_per_body = torch.matmul(body_embeds, title_embeds.t()) * logit_scale
        logits_per_title = logits_per_body.T

        loss = clip_loss(logits_per_body)

        return BTOutputs(
            loss=loss,
            logits_per_body=logits_per_body,
            logits_per_title=logits_per_title,
            body_embeds=body_embeds,
            title_embeds=title_embeds,
            body_outputs=body_outputs,
            title_outputs=title_outputs
        )
    def forward(self,
                input_ids_title,
                input_ids_body,
                attention_mask_title,
                attention_mask_body,
                pixel_values,
                global_attention_mask,
                ):
        
            
        title_outputs = self.clipModel.text_model(input_ids=input_ids_title, attention_mask=attention_mask_title)
        body_outputs = self.bodyModel(input_ids=input_ids_body, attention_mask=attention_mask_body,
                                      global_attention_mask=global_attention_mask)
        vision_outputs = self.clipModel.vision_model(pixel_values=pixel_values)
        TIOutput = self.clipModel(input_ids=input_ids_title, attention_mask=attention_mask_title,
                                  pixel_values=pixel_values)

        BIOutputs = self.multiple_BIoutput(body_outputs, vision_outputs)
        BTOutputs = self.multiple_BToutput(body_outputs, title_outputs)

        return LongClipOutput(
            BIoutputs=BIOutputs,
            BToutputs=BTOutputs,
            TIoutputs=TIOutput
        )



