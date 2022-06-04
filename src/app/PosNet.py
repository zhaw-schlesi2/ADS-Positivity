import logging
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertModel

class PosNet(nn.Module):
    def __init__(self,bertModel,trainBERT=False,outChannels=1,initialize=True):
        '''
        bertModel - The name or path of an existing bert model for sentence classification
        trainBERT - [False] Normally we do not train BERT for performance and quality reasons, it can be done if set to true. You may want to reduce the learning rate significantly and only train for a few epochs.
        outChannels -[1] Amount of outputs produced by the model
        initialize - [True]
        '''
        super().__init__()
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        
        self.bert = BertModel.from_pretrained(
            bertModel,
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1
        )
        
        #Disable training for the BERT model
        if(not trainBERT):
            self.disableBERTTraining()
        

        self.lstm = nn.LSTM(
            batch_first=True,
            num_layers=3,
            input_size=8, #8 tokens per step
            hidden_size=64, #64 timesteps
            dropout=0.15,
            bidirectional=True
        )
        self.lstmGELU = nn.GELU()
        self.lstmDropout = nn.Dropout(0.15)
        self.lstmDense = nn.Linear(64*128,self.bert.config.hidden_size)
        
        self.model = nn.Sequential(
            #nn.Linear(self.bert.config.hidden_size,self.bert.config.hidden_size),
            #nn.GELU(),
            #nn.Dropout(0.15),
            #nn.Linear(self.bert.config.hidden_size,outChannels)
            
            nn.Linear(self.bert.config.hidden_size*2,self.bert.config.hidden_size*2),
            nn.GELU(), #The outputs are in the negative range too, so use something that supports it.
            nn.Dropout(0.15),
            nn.Linear(self.bert.config.hidden_size*2,self.bert.config.hidden_size*2),
            nn.GELU(), 
            nn.Dropout(0.15),
            nn.Linear(self.bert.config.hidden_size*2,self.bert.config.hidden_size*2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.bert.config.hidden_size*2,outChannels),
        )
        if(initialize):
            self.model.apply(self.initialize)
            self.initialize(self.lstmDense)
    
    def disableBERTTraining(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.hidden_dropout_prob = 0.0
        self.bert.attention_probs_dropout_prob = 0.0
    
    def initialize(self,module):
        '''
        Initializes the weights of the layers
        '''
        if(isinstance(module,nn.Linear)):
            nn.init.kaiming_uniform_(module.weight)
            module.bias.data.fill_(0.0)
        elif(isinstance(module,nn.LSTM)):
            nn.init.kaiming_uniform_(module.weight)
            module.bias.data.fill_(0.0)
    
    def forward(self,inputs,mask,lstmInputs):
        #Sentence classification
        embeddings, pooledOut = self.bert(input_ids=inputs,attention_mask=mask,return_dict=False)
        lstmOut,lstmHidden = self.lstm(lstmInputs)
        x = self.lstmDropout(self.lstmGELU(lstmOut))
        x = x.reshape(inputs.size()[0],x.size()[1]*x.size()[2])#reshape to equal output size
        x = self.lstmDense(x)
        
        #Merge both results and pass to the rest of the model
        #-----
        #BERT uses transformers but previous techniques involved LSTMs which work as well.
        #The idea was to combine them and it seems to work better with the result that it did do better than either alone
        merged = torch.cat((pooledOut,x),dim=1).to(inputs.device).float()        
        return self.model(merged) 
    
    