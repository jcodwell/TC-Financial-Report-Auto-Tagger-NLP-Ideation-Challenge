import os
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
from pathlib import Path 



class Tokenizer():

    # init method or constructor
    def __init__(self):
       
        self.cwd = os.getcwd()
    # Train a tokenizer on Data
    def train_tokenizer(self):
        paths= [str(x) for x in Path(os.getcwd() +'/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/text/').glob('*.txt') ]
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=15_000, min_frequency=4, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

        os.makedirs('tokenizer_model')
        tokenizer.save_model('tokenizer_model')
        return "Finished Training..."
        
    # See tokenizer Output
    def test_tokenizer(self):
        # initialize the tokenizer using the tokenizer we initialized and saved to file
        tokenizer = RobertaTokenizer.from_pretrained('tokenizer_model', max_len=512)
        tokens = tokenizer('dei:DocumentAnnualReport')
        print(tokens)

        
# T =  Tokenizer()
# T.test_tokenizer()





