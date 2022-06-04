from transformers import BertTokenizer
import logging

class Tokenizer():
    '''
    Converts text to tokens and back
    '''
    def __init__(self,bertModel):
        '''
        bertModel - Either the name from a desired BERT model or, if not using a pre-trained one, the path to such a model.
        '''
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        self.log.info("Loading BERT tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(bertModel)
        
    def tokenize(self,text,padding="max_length",max_length=512,truncation=True,return_tensors="pt"):
        '''
        Just a wrapper for ease of use
        '''
        
        #https://arxiv.org/abs/1905.05583 - they found cutting out the text in the middle is more effective
        #An alternate may be Longformer models
        if(truncation):
            words = text.split(" ")
            if(len(words) > max_length):
                half = max_length//2
                a = words[:half]
                b = words[half*-1:]
                text=" ".join(a)+" ".join(b)
        
        return self.tokenizer(text,padding=padding, max_length=max_length, truncation=truncation, return_tensors=return_tensors)

    def decode(self,tokenIDs):
        '''
        Just a wrapper for ease of use
        '''
        return self.tokenizer.decode(tokenIDs)
    
    def getBackendTokenizer(self):
        '''
        Returns the actual tokenizer.
        '''
        return self.tokenizer