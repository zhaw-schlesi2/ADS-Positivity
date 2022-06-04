import zmq
import logging
import json

class PosZMQClient():
    def __init__(self):
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
    
    def send(self,zeroMQ,text):
        '''
        Sends a message to the ZeroMQ server of this toolset and prints the response
        zeroMQ - The address to connect to - check the zmq documentation
        text - Some review like text
        '''
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(zeroMQ)
        except Exception as e:
            self.log.error(f"Couldn't connect to {zeroMQ}")
            raise e
        
        self.log.info("Connected, sending message and waiting for reply")
        try:
            socket.send_string(text)
            response = socket.recv()
            rspObj = json.loads(response)
            rspCode = rspObj.get("rspCode")
            errorMsg = rspObj.get("errorMsg")
            if(not rspCode == 0):
                self.log.error(f"The server responded with an error code: {rspCode} - {errorMsg}")
                return
        except Exception as ex:
            self.log.error("Something went wrong during the data transfer with the server")
            raise ex
        
        reviewRounded = rspObj.get("rating")
        #reviewExact = rspObj.get("exactRating")
        reviewText = rspObj.get("text")
        out = "\n"*100
        out += f".------------------------.\n"
        out += f"| Response".ljust(25)+"|\n"
        out +=f"-------------------------|\n| TEXT START\n\\\n\n"
        out +=f"{reviewText}\n\n/\n| TEXT END\n"
        out +=f"|------------------------|\n"
        out+="| Rating:".ljust(20)+f"{str(reviewRounded).ljust(5)}|\n"
        out+="'------------------------'"+("\n"*2)
        print(out)