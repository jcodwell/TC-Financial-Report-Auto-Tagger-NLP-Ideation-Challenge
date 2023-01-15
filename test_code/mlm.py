import torch
import os
from pathlib import Path 
from tqdm.auto import tqdm 
from transformers import RobertaTokenizer

class MLM():

    # init method or constructor
    def __init__(self):
       self.txt_files = [str(x) for x in Path(os.getcwd() +'/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/text/').glob('*.txt') ]
       self.tokenizer = RobertaTokenizer.from_pretrained('tokenizer_model', max_len=512)
       self.encodings ={
        'input_ids': '',
        'attention_mask': '',
         'labels': ''
       }
       
    # Train a tokenizer on Data
    def mlm(self, tensor):
        rand = torch.rand(tensor.shape)
        # Tensor should not include special tokens 0,1,2
        mask_arr = (rand < 0.15) * (tensor != 0) * (tensor != 1) * (tensor != 2) 

        for i in range(tensor.shape[0]):
            #Returns a list of nonzeroes -> [[2,43,4]]
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            tensor[i, selection] = 4
        return tensor

    def create_tensor_attributes(self):
        input_ids = []
        mask = []
        labels =[]
        encodings = self.encodings
        for path in tqdm(self.txt_files):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
            sample = self.tokenizer(lines, max_length=512, padding='max_length', truncation=True)
            labels.append(torch.tensor(sample.input_ids)) 
            mask.append(torch.tensor(sample.attention_mask))
            input_ids.append(self.mlm(torch.tensor(sample.input_ids)))
        encodings['input_ids'] =  torch.cat(input_ids)
        encodings['attention_mask'] =  torch.cat(mask)
        encodings['labels'] =  torch.cat(labels)
        return encodings
        
 

