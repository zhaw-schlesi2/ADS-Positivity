import logging
import traceback
import numpy as np
import time
import torch
import torch.nn as nn
import os
from copy import copy
from torch.utils.data.dataloader import DataLoader
from src.app.ReviewDataSet import ReviewDataSet
from src.app.PosNet import PosNet
from torch.utils.tensorboard.writer import SummaryWriter


class PosApp():
    def __init__(self,datasetDir, tokenizerName, lazyCache, batchsize, learningRate, threshold, patience, totalEpochs, modelFile, update, trainBERT,humanBias):
        '''
        datasetDir - Directory that contains the training data (as the collector part of this tool stores them).
        tokenizerName - This string is passed to the Tokenizer implementation which could be extended in the future. For now you may only use a BERT model name (will be downloaded) or a path to a downloaded and pretrained BERT model. 
        lazyCache - If true, cache the processed data whenever an item is requested for faster access
        learningRate - The starting learning rate. This will be reduced under given circumstances
        threshold - The minimum positive change to reset the patience countdown to its original state
        patience - How many epochs without a positive improvement have to pass before the learning rate will be reduced.
        totalEpochs - if -1 the model will train infinitely. Saves happen whenever the model improves. Otherwise training is stopped at the given epoch. The best iteration will be saved.
        modelFile - The target file to store the model
        update - if true, will update an existing model
        trainBert - Train the bert model as well.
        humanBias - boost weights based on some numbers people tend to prefer over others.
        '''
        
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        self.log.info("Checking Cuda/ROCm support")
        self.datasetDir=datasetDir
        self.tokenizerName=tokenizerName
        self.lazyCache=lazyCache
        self.batchsize=batchsize
        self.learningRate=learningRate
        self.threshold=threshold
        self.patience=patience
        self.totalEpochs=totalEpochs
        self.modelFile=modelFile
        self.update=update
        self.trainBERT=trainBERT
        self.humanBias=humanBias
        self.setupDevice()
        
        
        #throws exception on errors during training.
        torch.autograd.set_detect_anomaly(True)
        
        
        #Load the datasets into the dataloaders for training and validation
        self.trainDS = ReviewDataSet(self.datasetDir,self.tokenizerName,self.lazyCache,self.humanBias)
        self.trainDS.setForTraining()
        
        self.validationDS = copy(self.trainDS)#it's faster and works. DataLoader might mess up the switching (untested)
        self.validationDS.setForValidation()
        
        self.trainDL = DataLoader(
            self.trainDS,
            batch_size=self.batchsize,
            shuffle=True,
            #num_workers=2,
            #persistent_workers=2
        )
        self.validationDL = DataLoader(
            self.validationDS,
            batch_size=self.batchsize,
            shuffle=True,
            #num_workers=2,
            #persistent_workers=2
        )
        
        #Those metrics are used for console output as well as storage for some internal values. Careful changing this.
        self.metrics = {
            "Model": self.modelFile,
            "Last Saved delta T": "n/a",
            "Last Saved delta E": "n/a",
            "Last Saved AVG Loss": "n/a",
            "Training Data": self.datasetDir,
            "SEP0":None,
            "Total Epochs": np.inf if self.totalEpochs == -1 else self.totalEpochs,
            "Current Epoch": 0,
            "Total Batches": 0,
            "Current Batch": 0,
            "Mode":"TRAINING",
            "SEP1":None,
            "Learning Rate": self.learningRate,
            "Threshold": self.threshold,
            "Patience": self.patience,
            "SEP2":None,
            "T Loss": np.inf,
            "T Loss AVG": np.inf,
            "T Accuracy": 0.0,
            "T Accuracy AVG": 0.0,
            "SEP3":None,
            "V Loss": np.inf,
            "V Loss AVG": np.inf,
            "V Accuracy": 0.0,
            "V Accuracy AVG": 0.0
        }
        
        self.startTraining()
        
    @staticmethod
    def printMetrics(metrics,dbgOut=None):
        '''
        Print the metrics that are kept up to date by the training loop.
        metrics - The current state of the metrics
        dbgOut - A tuple of (model outputs, targets). Up to 5 samples, depending on batchsize will be displayed in realtime if given.
        '''
            
        out = ""
        if(os.environ.get("DEBUG") == "y" and dbgOut):
            outputs,targets = dbgOut
            maxSamples = 5 if len(targets) > 5 else len(targets)
            out+="\n| DEBUG SAMPLE OUTPUT:\n"
            out+=("|"+("-"*104))+".\n"
            for batch in range(maxSamples):
                bstr = str(batch).rjust(4)
                out+=f"| Output[{bstr}]  {str(outputs[batch].item()).rjust(25)} == {str(targets[batch].item()).ljust(25)} Target[{bstr}]".ljust(105)+"|\n"
            out+=("|"+("'"*104))+"|\n"
        else:
            out+="\n"*9
            
        out+="| METRICS:".ljust(105)+"|\n"
        out+=("|"+("-"*104))+"|\n"
        for key,value in metrics.items():
            if(key.startswith("SEP")):
                out+="|"+("-"*104)+"|\n"
                continue
            out+=f"| {key.ljust(20)} | {str(value).ljust(80)}|\n"
        out+="'"+("-"*104)+"'\n"
        out+="\033[F"*(out.count("\n")+1)
        print(out)
        
    def startTraining(self):
        '''
        Starts the main training loop
        '''
        
        #Model initialization, most tweaks go here.
        self.log.info("Initialize model!")
        model = PosNet(bertModel=self.tokenizerName,outChannels=1,trainBERT=self.trainBERT).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learningRate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode="abs"
        )
        lossFunction = PosApp.weightedSmoothL1Loss #Performs better than MSELoss
        #lossFunctionInit = nn.BCEWithLogitsLoss# <-reinitializing this later as recommended by the docs for weights.
        #lossReduction = "mean"
        
        accFunction = PosApp.accuracy
        
        
        #Load the states from an existing model when asked to resume training.
        checkpoint = None
        if(self.update):
            self.log.info(f"Updating existing model {self.modelFile}, loading states...")
            checkpoint = torch.load(self.modelFile,map_location=(self.device))#map to the right device (allows to train on gpu and run on cpu or vica versa)
            model.load_state_dict(checkpoint.get("modelStateDict"))
            optimizer.load_state_dict(checkpoint.get("optimizerStateDict"))
            scheduler.load_state_dict(checkpoint.get("schedulerStateDict"))
            if(not self.trainBERT):
                model.disableBERTTraining()
            #It wouldn't do much harm but messes up the statistics about the model, refuse this
            if(self.metrics.get("Total Epochs") <= checkpoint.get("currentEpoch")):
                self.log.error(f"Please increase the number of target epochs beyond what was already trained [{checkpoint.get('currentEpoch')} Epochs] when updating the model")
            
        graph = SummaryWriter()
        
        self.log.info("Starting training!\n")
        self.log.info("-"*50)
        
        epoch = 0 if not checkpoint else checkpoint.get("currentEpoch")
        lastSaved = 0 if not checkpoint else time.time()#don't immediately save it when updating. Assume it was saved when loaded.
        lastSavedE = epoch
        lastAVGValidationLoss = np.inf if not checkpoint else checkpoint.get("minTargetValidationLoss")
        graphTimeStepTraining = 0
        graphTimeStepValidation = 0
        print("\n"*1000, end="")#clear screen
        while(True):
            PosApp.printMetrics(self.metrics)
            tLossSUM = 0.0
            tAccSUM = 0.0
            vLossSUM = 0.0
            vAccSUM = 0.0
            epoch+=1
            self.metrics.update({"Current Epoch":epoch})
            self.metrics.update({"Last Saved delta E":epoch-lastSavedE})
            
            #----------- Training -----------
            #
            model.train()
            batch = 0
            self.metrics.update({"Mode":"TRAINING", "Total Batches": len(self.trainDL)})
            for args in self.trainDL:
                graphTimeStepTraining+=1
                batch+=1
                self.metrics.update({"Current Batch":batch})
                
                inputs,targets,mask,weights,lstmInputs = args
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                weights = weights.to(self.device)
                lstmInputs = lstmInputs.to(self.device)
                
                optimizer.zero_grad()
                out = model(inputs,mask,lstmInputs)
                
                #lossFunction = lossFunctionInit(reduction=lossReduction,weight=weights)
                loss = lossFunction(out,targets,weights)
                loss.backward()
                self.metrics.update({"T Loss":loss.item()})
                tLossSUM+=loss.item()
                
                acc = accFunction(out,targets).item()
                self.metrics.update({"T Accuracy":acc})
                tAccSUM+=acc
                optimizer.step()
                
                graph.add_scalar("Training Loss",loss,graphTimeStepTraining)
                graph.add_scalar("Training Accuracy",acc,graphTimeStepTraining)
                PosApp.printMetrics(self.metrics,dbgOut=(out,targets))
                saveDelta = time.time() - lastSaved
                self.metrics.update({"Last Saved delta T":int(saveDelta)})
                
            self.metrics.update({"T Loss AVG":tLossSUM/len(self.trainDL)})
            self.metrics.update({"T Accuracy AVG":tAccSUM/len(self.trainDL)})
            graph.add_scalar("Training Loss per Epoch",self.metrics.get("T Loss AVG"),epoch)
            graph.add_scalar("Training Accuracy per Epoch",self.metrics.get("T Accuracy AVG"),epoch)
            PosApp.printMetrics(self.metrics)
            
            #
            #---------- Validation ----------
            #
            model.eval()
            batch = 0
            self.metrics.update({"Mode":"EVALUATION", "Total Batches": len(self.validationDL)})
            for args in self.validationDL:
                graphTimeStepValidation+=1
                batch+=1
                self.metrics.update({"Current Batch":batch})
                
                with torch.no_grad():
                    inputs,targets,mask,weights,lstmInputs = args
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    mask = mask.to(self.device)
                    weights = weights.to(self.device)
                    lstmInputs = lstmInputs.to(self.device)
                    
                    out = model(inputs,mask,lstmInputs)
                    
                    #lossFunction = lossFunctionInit(reduction=lossReduction,weight=weights)
                    loss = lossFunction(out,targets,weights)
                    self.metrics.update({"V Loss":loss.item()})
                    vLossSUM+=loss.item()
                    
                    acc = accFunction(out,targets).item()
                    self.metrics.update({"V Accuracy":acc})
                    vAccSUM+=acc
                
                graph.add_scalar("Validation Loss",loss,graphTimeStepValidation)
                graph.add_scalar("Validation Accuracy",acc,graphTimeStepValidation)
                PosApp.printMetrics(self.metrics,dbgOut=(out,targets))
                saveDelta = time.time() - lastSaved
                self.metrics.update({"Last Saved delta T":int(saveDelta)})
                                
            self.metrics.update({"V Loss AVG":vLossSUM/len(self.validationDL)})
            self.metrics.update({"V Accuracy AVG":vAccSUM/len(self.validationDL)})
            graph.add_scalar("Validation Loss per Epoch",self.metrics.get("V Loss AVG"),epoch)
            graph.add_scalar("Validation Accuracy per Epoch",self.metrics.get("V Accuracy AVG"),epoch)
            PosApp.printMetrics(self.metrics)
            #
            #-------- Before next epoch --------
            
            scheduler.step(self.metrics.get("V Loss AVG"))
            
            #Saving the model when it performs better than a previous version and at least 60 seconds have passed or the last epoch was reached.
            saveDelta = time.time() - lastSaved
            self.metrics.update({"Last Saved delta T":int(saveDelta)})
            if(saveDelta > 60 or epoch == self.metrics.get("Total Epochs")):#Don't save more often than every 60s unless it's the last epoch
                if(lastAVGValidationLoss > self.metrics.get("V Loss AVG")):
                    lastAVGValidationLoss = self.metrics.get("V Loss AVG")
                    lastSaved = time.time()
                    self.metrics.update({"Last Saved AVG Loss":lastAVGValidationLoss})
                    lastSavedE = epoch
                    torch.save({
                        "modelStateDict":model.state_dict(),
                        "optimizerStateDict":optimizer.state_dict(),
                        "schedulerStateDict":scheduler.state_dict(),
                        "currentEpoch":epoch,
                        "minTargetValidationLoss":lastAVGValidationLoss,
                        "graphTimeStepTraining":graphTimeStepTraining,
                        "graphTimeStepValidation":graphTimeStepValidation,
                        "lastSaved":lastSaved
                    },self.modelFile)
                    self.metrics.update({"Last Saved delta T":0})
                    PosApp.printMetrics(self.metrics)
                
    @staticmethod
    def accuracy(outputs,targets):
        '''
        Returns the accuracy of the given ouputs compared to targets (rounded to 2 decimals on the down scaled target -> 0-1 -> 0.01). 0-100%
        '''
        return 100*(torch.sum(torch.round(outputs,decimals=2) == torch.round(targets,decimals=2))/len(targets))
    
    @staticmethod
    def weightedMSELoss(outputs, targets, weights):
        '''
        Since we have a weighted dataset, apply those weights in the MSELoss function to prevent bias.
        This is the equivalent to MSELoss from pytorch just with the weights.
        '''
        return torch.mean(weights * (outputs - targets) ** 2)
    
    @staticmethod
    def weightedSmoothL1Loss(inputs, targets, weights):
        '''
        Since we have a weighted dataset, apply those weights in the SmoothL1Loss function to prevent bias.
        This is the equivalent to SmoothL1Loss from pytorch just with the weights
        '''
        t = torch.abs(inputs - targets)
        return torch.mean(weights * torch.where(t < 1, 0.5 * t ** 2, t - 0.5))
            
    def setupDevice(self):
        '''
        Picks the most appropriate device available
        '''
        if(torch.cuda.is_available()):
            gpucount = torch.cuda.device_count()
            self.log.info(f"Found {gpucount} Cuda capable GPUs - Using the first one. You may change this by setting CUDA_VISIBLE_DEVICES=[n]")
            self.device = "cuda:0"
            try:
                rngArray = torch.rand(4, 3).to(self.device)
                if(not len(rngArray) == 4):
                    self.log.warn("For some unknown reason Cuda is not functional - fallback to cpu!")
                    self.device = "cpu"
                for i in range(0,4):
                    if(not len(rngArray[i]) == 3):
                        self.log.warn("For some unknown reason Cuda is not functional - fallback to cpu!")
                        self.device = "cpu"
                        break
            except Exception:
                traceback = traceback.format_exc()
                self.log.error("Couldn't copy test array to gpu: "+str(traceback))
                self.log.warn("fallback to cpu")
                self.device = "cpu"
        else:
            self.device = "cpu"