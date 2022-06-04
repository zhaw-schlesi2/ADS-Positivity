import logging
import torch
import traceback
import zmq
import json
from src.app.PosNet import PosNet
from src.app.Tokenizer import Tokenizer
from src.app.ReviewDataSet import ReviewDataSet


class PosSentiment():
    '''
    Provides an interface to interact with the trained model
    '''
    
    def __init__(self, zeroMQ, tokenizerNameOrPath, modelpath):
        '''
        zeroMQ - A string telling the zeroMQ Server where to listen, check their documentation for further information.
        tokenizerNameOrPath - The name or path of the tokenizer that was used during training(!)
        modelpath - The path to the trained model
        '''
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        self.setupDevice()
        self.modelpath = modelpath
        self.zeroMQ = zeroMQ
        self.tokenizerNameOrPath = tokenizerNameOrPath
        self.tk = Tokenizer(tokenizerNameOrPath)
        
        
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
                tb = traceback.format_exc()
                self.log.error("Couldn't copy test array to gpu: "+str(tb))
                self.log.warn("fallback to cpu")
                self.device = "cpu"
        else:
            self.device = "cpu"
    
    def processInput(self,text):
        '''
        text - Tokenize and prepare a given text to send to the model
        '''
        tkData = self.tk.tokenize(text)
        inputs = torch.tensor(tkData.get("input_ids")).to(self.device)[0]
        lstmInputs = torch.flip(inputs,(0,))
        lstmInputs = lstmInputs.reshape(lstmInputs.size()[0]//8,8).float().to(self.device)
        mask = torch.tensor(tkData.get("attention_mask")).to(self.device)[0]
        return (torch.unsqueeze(inputs,0),torch.unsqueeze(mask,0),torch.unsqueeze(lstmInputs,0))
    
    
    def startZMQ(self):
        '''
        Starts a ZMQ server and listens/responds to requests
        '''
        self.log.info(f"Initialize model...")
        model = PosNet(bertModel=self.tokenizerNameOrPath,outChannels=1).to(self.device)
        self.log.info(f"Loading last checkpoint from {self.modelpath}...")
        checkpoint = torch.load(self.modelpath,map_location=(self.device))#map to the right device (allows to train on gpu and run on cpu or vica versa)
        model.load_state_dict(checkpoint.get("modelStateDict"))
        model.eval()
        self.log.info(f"Model ready!")
        
        self.log.info(f"Starting ZMQ server {self.zeroMQ}...")
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.zeroMQ)
        self.log.info(f"ZMQ ready! Listening now\n\n\n>>> (waiting for client data on {self.zeroMQ})")
        
        while(True):
            try:
                msg = socket.recv().decode("utf-8")
                msg = msg.replace("\n","").replace("\r\n","")
                self.log.info("Received message")
                out = "\n"*2
                out += f".------------------------.\n"
                out += f"| Message".ljust(25)+"|\n"
                out += f"-------------------------|\n| TEXT START\n\\\n\n"
                out +=f"{msg}\n\n/\n| TEXT END\n"
                with torch.no_grad():
                    inputs,masks,lstmInputs=self.processInput(msg)
                    output = model(inputs,masks,lstmInputs)*ReviewDataSet.maxRating
                    outputRounded = round(output.item(),1)
                socket.send_string(json.dumps({
                    "text":msg,
                    "rating":outputRounded,
                    "exactRating":output.item(),
                    "maxRating":ReviewDataSet.maxRating,
                    "minRating":ReviewDataSet.minRating,
                    "rspCode":0,
                    "errorMsg":None
                }))
                out +=f"|------------------------|\n"
                out+="| Rating:".ljust(20)+f"{str(outputRounded).ljust(5)}|\n"
                out+="'------------------------'"+("\n"*2)
                print(out)
                
            except KeyboardInterrupt:
                self.log.info("stopped ZQM")
                break
            except Exception as e:
                try:
                    socket.send_string(json.dumps({
                        "rspCode":1,
                        "errorMsg": str(traceback.format_exc())
                    }))
                finally:
                    tb = traceback.format_exc()
                    self.log.error(tb)
                    continue
            
            self.log.info("Waiting for next message")
        
        
        