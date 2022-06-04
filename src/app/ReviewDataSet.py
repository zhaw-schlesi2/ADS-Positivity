from torch.utils.data import Dataset
import math
import logging
import json
import os
import torch
from json.decoder import JSONDecodeError
from src.app.Tokenizer import Tokenizer

class ReviewDataSet(Dataset):
    '''
    Loads reviews from one or more jsonl file and provides inputs/targets/masks for ML.
    '''
    
    minRating = 0.0
    maxRating = 10.0
    #TODO: add step size - ratings are rounded to .5 as of now for weight distribution.

    def __init__(self, datasetDir, tokenizerName, lazyCache=False, humanBias=True):
        '''
        datasetDir - Directory that contains the jsonl files with the reviews.
        tokenizerName - Path or name of a tokenizer that is implemented.
        lazyCache - Store processed entries in ram when requested for faster access when requested again.
        humanBias - boost weights cause people tend to have a bias towards certain numbers
        '''
        super().__init__()
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        self.datasetDir = datasetDir
        self.lazyCache=lazyCache
        self.humanBias=humanBias
        
        self.tk = Tokenizer(tokenizerName)
        
        self.validation = False
        self.training = False
        
        self.index = []
        self.data = []
        self.ratingDistribution = {}
        self.ratingWeight = {}
        files = []
        
        for filename in os.listdir(self.datasetDir):
            ext = os.path.splitext(filename)[1]
            filepath = os.path.join(self.datasetDir,filename)
            if(ext != ".jsonl"):
                self.log.warn(f"Skipping file: {filepath} - doesn't have the jsonl extension.")
                continue
            self.log.info(f"Indexing file: {filepath}")
            files.append(filepath)
        
        currentFileIDX = 0
        
        tReviews = 0
        for filepath in files:
            currentFileIDX+=1
            cFilename = os.path.basename(filepath)
            
            reader = open(filepath,"r",1)
            reader.seek(0)
            currentLine = 0
            while(True):
                start = reader.tell()
                line = reader.readline()
                bytesRead = len(line)
                currentLine+=1
                if(len(line.strip()) <= 2):
                    #print(f"\n'->EOF at line: {currentLine} ")
                    print()
                    break

                try:
                    review = json.loads(line)
                except JSONDecodeError as jde:
                    self.log.warn(f"Skipping line - JSON decode error on line {currentLine} in {filepath}")
                    continue
                
                #TODO: Add filters here if needed - call continue
                #if(review.get("helpful") < 0.6):
                #    continue
                if(len(review.get("text")) < 128):#filter too short reviews
                    continue
                
                tReviews+=1
                print(f"Reading data - [File {currentFileIDX}/{len(files)}] [Total reviews {tReviews}] [Reading: {cFilename}]", end = "\r")
                
                 
                #Count rating distribution to adjust weights later
                ratingRounded = round(review.get("rating")*2)/2
                currentRatingCount = self.ratingDistribution.get(str(ratingRounded))
                if(not currentRatingCount):
                    self.ratingDistribution.update(
                        {str(ratingRounded):1}
                    )
                else:
                    self.ratingDistribution.update(
                        {str(ratingRounded):currentRatingCount+1}
                    )
                
                self.index.append({
                    "reader":reader,
                    "seek":start,
                    "length":bytesRead
                })
                if(self.lazyCache):
                    self.data.append(None)
        if(len(self.index) == 0):
            raise RuntimeError("No valid data found!")
        self.__processWeights()
        self.__printWeights()
    
    def setForValidation(self):
        '''
        Limits the dataset to 1/5 of every entry
        '''
        self.validation=True
        self.training=False
    
    def setForTraining(self):
        '''
        Limits the dataset to 4/5 of every entry
        '''
        self.validation=False
        self.training=True
        
    
    def __processWeights(self):
        '''
        Creates a lookup table for rounded ratings (0.5) to adjust the weights
        '''
        self.log.info("Creating weight lookup table")
        cRating = ReviewDataSet.minRating
        total = len(self.index)#one rating per review
        maxCount = 0
        
        for key,value in self.ratingDistribution.items():
            if(maxCount < value):
                maxCount = value
        
        while(True):
            if(cRating > ReviewDataSet.maxRating+1e-8):#+eps
                break
            ratingCount = self.ratingDistribution.get(str(cRating))
            if(not ratingCount):
                cRating+=0.5
                continue
            percentage = (maxCount/ratingCount)
            self.ratingWeight.update({str(cRating):percentage})
            cRating+=0.5

        if(self.humanBias):
            if(ReviewDataSet.maxRating == 10.0 and ReviewDataSet.minRating == 0.0):
                self.ratingWeight.update({
                    str("10.0"):self.ratingWeight.get("10.0")+0.5,
                    str("1.0"):self.ratingWeight.get("1.0")+0.5
                })
            if(ReviewDataSet.maxRating == 5.0 and ReviewDataSet.minRating == 0.0):
                self.ratingWeight.update({
                    str("5.0"):self.ratingWeight.get("5.0")+0.5,
                    str("1.0"):self.ratingWeight.get("1.0")+0.5
                })

            
    def __printWeights(self):
        #if(not os.environ.get("DEBUG") == "y"):
        #    return
        print(f"\n\nWEIGHTS FOR DATASET (Human Bias: {self.humanBias})")
        print("-----------------------------------------")
        for key,value in self.ratingWeight.items():
            print(f"Rating: {str(float(key)).rjust(4)} | Weight: {value}")
        print("-----------------------------------------")
        for key,value in self.ratingDistribution.items():
            print(f"Rating: {str(float(key)).rjust(4)} | Count : {value}")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    
    def __processData(self,obj):
        '''
        processes the review and returns (inputs,targets,mask) as tensors
        '''
        tkData = self.tk.tokenize(obj.get("text"))
        inputs = tkData.get("input_ids")[0]
        lstmInputs = torch.flip(inputs,(0,))
        lstmInputs = lstmInputs.reshape(lstmInputs.size()[0]//8,8).float()
        mask = tkData.get("attention_mask")[0]
        targets = torch.tensor([obj.get("rating")/ReviewDataSet.maxRating])#Scale it down to 0-1
        weight = torch.tensor([self.ratingWeight.get(str(round(obj.get("rating")*2)/2))])
        return inputs,targets,mask,weight,lstmInputs
    
    def __getIndex(self,idx):
        '''
        Return different elements based on current dataset mode
        '''
        if(self.validation):
            idx = idx*5
        if(self.training):
            idx = idx+1+int(math.floor(idx/4))
        return idx
    
    def __len__(self):
        '''
        Return different length based on current dataset mode
        '''
        total = len(self.index)
        if(self.validation):# 1/5th of the training set
            return int(math.ceil(total*0.2))
        if(self.training):# 4/5th of the training set
            return int(math.floor(total*0.8))
        return len(self.index)

    def __getitem__(self, index):
        '''
        index - Returns the item at index position.
        '''
        if(index > self.__len__()):
            raise StopIteration
        index = self.__getIndex(index)
        
        if(self.lazyCache):
            #when caching is enabled, get start and end position, read, process and cache the review. 
            if(not self.data[index]):
                reader = self.index[index].get("reader")
                reader.seek(self.index[index].get("seek"))
                review = json.loads(reader.read(self.index[index].get("length")))
                entry = self.__processData(review)
                self.data[index] = entry
            return self.data[index]
        
        #else process it each time.
        reader = self.index[index].get("reader")
        reader.seek(self.index[index].get("seek"))
        review = json.loads(reader.read(self.index[index].get("length")))
        return self.__processData(review)
        
        

