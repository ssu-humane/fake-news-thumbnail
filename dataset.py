import numpy as np
from transformers import CLIPProcessor, LongformerTokenizer
import torch
import pandas as pd
from PIL import Image
import requests
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

def context_padding(inputs, context_length=77):
    shape = (1,context_length - inputs.input_ids.shape[1])
    x = torch.zeros(shape)
    input_ids = torch.cat([inputs.input_ids,x], dim=1).long()
    attention_mask = torch.cat([inputs.attention_mask,x], dim=1).long()
    return input_ids, attention_mask
def context_padding_longformer(input_id, context_length=4096): #4096 token 까지만 받을 수 있음
    if input_id.shape[1] > context_length:
        input_ids = torch.cat([input_id[:,:context_length-1],torch.Tensor([[2]]) ], dim=1).long()
    elif input_id.shape[1] < context_length:
        shape = (1,context_length - input_id.shape[1])
        x = torch.zeros(shape)
        input_ids = torch.cat([input_id,x], dim=1).long()
    else:
        input_ids = input_id
    return input_ids

class MyDataset(Dataset):
    def __init__(self, df, images_path):
        super().__init__()
        self.df = df
        self.images_path = images_path
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    def __len__(self):
        return len(self.df)
    def slicing_ids(self, body_input_id, words_length):

        paragraph = []
        for id_ in body_input_id:
            if id_ == 50140 or id_ == 2 or id_ == 50118: #\n = 50118, \n\n = 50140
                break
            paragraph.append(id_)

        body_output_id = body_input_id[len(paragraph)+1:]
        if body_output_id[0] == 50118:
            body_output_id = body_output_id[1:]
        print(body_output_id)
        return body_output_id
    
    def padding_truncate_ids(self, input_id,words_length):

        output_id = []
        #넘칠 경우 자르기
        if len(input_id) > words_length:
            output_id = input_id[:words_length]
        #부족할 경우 1(pad_token_id) 만큼 채워주기   
        elif len(input_id) < words_length:
            output_id = input_id
            for _ in range(words_length - len(input_id)):
                output_id.append(1) #pad_token_id : 1
        else:
            output_id = input_id
        print("processed output: ",output_id)
        print('결과 길이: ', len(output_id))
        return output_id

    def get_body_processing(self, body_input_id, paragraph_cnt, words_length = 50):

        enter_token = 0 #<s>
        paragraphs = []
        #첫 토큰은 빼놓음
        #body_input_id = list
        body_input_id2 = body_input_id[1:]

        #실제 문단 개수
        real_paragraph_cnt = 0
        for idx, id_ in enumerate(body_input_id2):
            if id_ == 50118:
                if body_input_id2[idx-1] == 50118:
                    real_paragraph_cnt += 1
            elif id_ == 50140:
                real_paragraph_cnt += 1
        
        for cnt in range(paragraph_cnt):
            if cnt < real_paragraph_cnt:
                body_input_id2 = self.slicing_ids(body_input_id2, words_length)
            #실제 문단이 원하는 문단 개수보다 작다면 나머지 문단들은 padding token 으로 채운다.
            else:
                body_input_id2 = []

            output_id = self.padding_truncate_ids(body_input_id2, words_length)
            paragraphs += output_id
            paragraphs += [enter_token]
        paragraphs[-1] = 2  #end_token
        return [0] + paragraphs
        
    def check_id_to_string(self, input_ids):
        outputs = []
        for id_ in input_ids:
            outputs += self.tokenizer.convert_tokens_to_string(self.tokenizer._convert_id_to_token(id_))
        #print(*outputs)

    def remove_sequential_enter(self, body):
        #\n\n\n\n\n 같은 부분이 있어서 \n\n로 만드는 과정
        text = body.split('\n\n')
        while '' in text:
            text.remove('')
        body_output = ''
        #body 다시 원래대로 만들기
        for paragraph in text:
            body_output += paragraph
            body_output += '\n\n'
        #마지막은 \n\n 없앰
        body_output = body_output[:-2]

        return body_output

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        print(row['id'])
        image = Image.open(f"{self.images_path}/{row['id']}/meta_img.png").convert("RGB")
        #img_url = row['image']
        #image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        body = self.remove_sequential_enter(row['body'])
        # body = body.split('\n\n')

        inputs = self.processor(text=row['title'], images=image, return_tensors="pt", padding=True)
        title_input_id, attention_mask = context_padding(inputs)

        body_input_id = self.tokenizer.encode(body)
        #body_input_id = self.tokenizer.encode(self.tokenizer.pad_token)
        #id 가 알맞게 잘 들어왔는지 확인하는 함수
        self.check_id_to_string(body_input_id)
        body_input_id = self.get_body_processing(body_input_id, paragraph_cnt=10, words_length=50)
        print(body_input_id)
        self.check_id_to_string(body_input_id)

        #기사들 분포 보기 문단 뒤쪽에다가 일저
        body_input_id = torch.tensor(body_input_id).unsqueeze(0)
        body_input_id = context_padding_longformer(body_input_id)

        attention_mask_body = torch.ones(body_input_id.shape, dtype=torch.long, device=body_input_id.device)
        global_attention_mask = torch.zeros(body_input_id.shape, dtype=torch.long,
                                            device=body_input_id.device)  # initialize to global attention to be deactivated for all tokens
        global_attention_mask[:, [0, 51, 102,153,204,255,306,357,408, 459]] = 1

        print('input_ids shape', inputs.input_ids.shape)
        print('attention_mask shape', inputs.attention_mask.shape)
        print('pixel_values shape', inputs.pixel_values.squeeze().shape)
        print('body_inputs input_ids shape', body_input_id.shape)
        print('global_attention_mask shape', global_attention_mask.shape)

        return title_input_id.squeeze (), body_input_id.squeeze(), attention_mask.squeeze(), attention_mask_body.squeeze(), inputs.pixel_values.squeeze(), global_attention_mask.squeeze()