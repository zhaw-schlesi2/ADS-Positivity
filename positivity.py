#!/usr/bin/env python3
import argparse
import logging
from src.app.ReviewDataSet import ReviewDataSet
from src.app.PosSentiment import PosSentiment
from src.app.PosZMQClient import PosZMQClient
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
import sys
import os
import traceback
import multiprocessing
import signal
from pathlib import Path
from src import paths
from src.app.Collector import Collector
from src.app.PosApp import PosApp
multiprocessing.freeze_support()

if __name__ == "__main__":

    log = logging.getLogger("Positivity")
    ap = argparse.ArgumentParser(description='Positivity')
    sp = ap.add_subparsers(help="Target")
    
    trainParser = sp.add_parser("train")
    trainParser.set_defaults(which="trainParser")
    trainParser.add_argument("-t","--train-dir",help=f"[{paths.APPDATA_SCRAPER}] The training set directory (jsonl files)",default=paths.APPDATA_SCRAPER)
    trainParser.add_argument("-c","--lazy-cache",help="[False] Cache the training data on item request",default=False,action="store_true")
    trainParser.add_argument("-x","--total-epochs",help="[-1] Total epochs. ",default=-1,type=int)
    trainParser.add_argument("-m","--model-path",help=f"[{os.path.join(paths.APPDATA_MODELS,paths.PROJECTNAME,'.pth')}] The path to save the trained model. If it exists it will be further trained.",default=os.path.join(paths.APPDATA_MODELS,'pth'))
    trainParser.add_argument("-b","--batch-size",help="[32] Training minibatch size",default=32,type=int)
    trainParser.add_argument("-u","--update",help="[No] update existing model",default=False,action="store_true")
    trainParser.add_argument("-v","--verbose",help="[No] set DEBUG environment variable",default=False,action="store_true")
    trainParser.add_argument("--no-human-bias",help="[No] Disables the weight boost added because humans tend to have a bias towards certain numbers.",action="store_false",default=True)
    trainParser.add_argument("--train-bert",help="[No] Train the bert model as well - this is not recommended. If you do, use 2-4 epochs with a reduced learning rate and then resume without this parameter.",default=False,action="store_true")
    trainParser.add_argument("--tokenizer",help="[bert-base-multilingual-cased] Name of a (supported) tokenizer or path if customized.",default="bert-base-multilingual-cased")
    trainParser.add_argument("--learning-rate",help="[1e-4] Start learning rate",default=1e-4,type=float)
    trainParser.add_argument("--threshold",help="[1e-3] Threshold for learning rate reduction if no improvement over n epochs was detected",default=1e-3,type=float)
    trainParser.add_argument("--patience",help="[5] Patience for threshold to be met. It that doesn't happen within n epochs, the learning rate is reduced",default=5,type=int)
    
    collectParser = sp.add_parser("collect")
    collectParser.set_defaults(which="collectParser")
    collectParser.add_argument("-k","--api-key",help="[REQUIRED] Your API key from IMDB",required=True)
    collectParser.add_argument("-t","--title-list",help="List of titles to fetch reviews about",nargs="+",default=[])
    collectParser.add_argument("-T","--title-list-file",help="Path to a file with movie titles - one per line")
    collectParser.add_argument("-d","--data-dir",help=f"[{paths.APPDATA_SCRAPER}] Directory to store the fetched reviews in",default=paths.APPDATA_SCRAPER)
    collectParser.add_argument("-p","--permissive-search",help="Fetch reviews from all found and not just the most relevant title", action="store_true")
    collectParser.add_argument("--minr",help="Minimum rating the reviews can have to be fetched",default=ReviewDataSet.minRating,type=float)
    collectParser.add_argument("--maxr",help="Maximum rating the reviews can have to be fetched",default=ReviewDataSet.maxRating,type=float)
    
    runParser = sp.add_parser("run")
    runParser.set_defaults(which="runParser")
    runParser.add_argument("-m","--model-path",help=f"[{os.path.join(paths.APPDATA_MODELS,paths.PROJECTNAME,'.pth')}] The path to the trained model.",default=os.path.join(paths.APPDATA_MODELS,paths.PROJECTNAME,'pth'))
    runParser.add_argument("-t","--tokenizer",help="[bert-base-multilingual-cased] Name/Path of the same(!) tokenizer used during training",default="bert-base-multilingual-cased")
    runParser.add_argument("-z","--zero-mq",help="[tcp://127.0.0.1:8888] The ZMQ \"protocol://ip:port\" to listen on for inputs.",default="tcp://127.0.0.1:8888")
    
    clientParser = sp.add_parser("client")
    clientParser.set_defaults(which="clientParser")
    clientParser.add_argument("-t","--text",help="Send any text to predict your rating",required=True)
    clientParser.add_argument("-z","--zero-mq",help="[tcp://127.0.0.1:8888] ZMQ Server to connect to",default="tcp://127.0.0.1:8888")
    
    if(len(sys.argv) <= 1):
        ap.print_help()
        exit(0)
    
    args = vars(ap.parse_args())

    def terminate(code=0):
        print("\n"*50)#quick hack because of runtime output cursor positon
        log.info("Terminating...")
        #TODO: shutdown threads
        if(code == 0):
            log.info(f"Terminated exit code:{code}")
        else:
            log.warning(f"Terminated exit code:{code}")
        exit(code)
        
    signal.signal(signal.SIGTERM, terminate)
        
    def existAndReturnOrExit(path):
        if(os.path.exists(path)):
            return path
        log.error(f"Required file/folder doesn't exist ({path})")
        terminate(1)
        
    if(args["which"] == "collectParser"):
        try:
            titleList = args.get("title_list")
            titleListFile = args.get("title_list_file")
            if(titleListFile):
                titleListFile = existAndReturnOrExit(titleListFile)
                with open(titleListFile,"r") as fr:
                    for title in fr:
                        if(len(title.strip())>0):
                            titleList.append(title.strip())
            datadir = existAndReturnOrExit(args.get("data_dir"))
            permissive = args.get("permissive_search")
            apikey = args.get("api_key")
            minr = args.get("minr")
            maxr = args.get("maxr")
            collector = Collector(apikey,targetdir=datadir,permissive=permissive,minRating=minr,maxRating=maxr)
            collector.searchAndFetch(titleList)
        except KeyboardInterrupt as e:
            terminate(0)
        except Exception:
            traceback = traceback.format_exc()
            log.error("Application stopped unexpectedly: "+str(traceback))
            terminate(1)
            
    if(args["which"] == "clientParser"):
        try:
            zmqClient = PosZMQClient()
            zeroMQ = args.get("zero_mq")
            text = args.get("text")
            zmqClient.send(zeroMQ,text)
        except KeyboardInterrupt as e:
            terminate(0)
        except Exception:
            traceback = traceback.format_exc()
            log.error("Application stopped unexpectedly: "+str(traceback))
            terminate(1)
    
    if(args["which"] == "runParser"):
        try:
            modelPath = existAndReturnOrExit(args.get("model_path"))
            tokenizer = args.get("tokenizer")
            zeroMQ = args.get("zero_mq")
            pSentiment = PosSentiment(zeroMQ,tokenizer,modelPath)
            pSentiment.startZMQ()
        except KeyboardInterrupt as e:
            terminate(0)
        except Exception:
            traceback = traceback.format_exc()
            log.error("Application stopped unexpectedly: "+str(traceback))
            terminate(1)
            
    if(args["which"] == "trainParser"):
        try:
            datasetDir = existAndReturnOrExit(args.get("train_dir"))            
            modelPath = Path(args.get("model_path"))
            update = args.get("update")
            if(os.path.isfile(modelPath) and not update):
                log.error(f"You need to set to update flag to update the existing model: {modelPath}")
                terminate(1)
            if(update and not os.path.isfile(modelPath)):
                log.error(f"You set the update flag but the model {modelPath} doesn't exist")
                terminate(1)
            if(not os.path.isdir(modelPath.parent.absolute())):
                log.error(f"The parent dir of the given save location {modelPath} doesn't exist")
                terminate(1)
            tokenizer = args.get("tokenizer")
            lazyCache = args.get("lazy_cache")
            minibatchSize = args.get("batch_size")
            tEpochs = args.get("total_epochs")
            learningRate = args.get("learning_rate")
            threshold = args.get("threshold")
            patience = args.get("patience")
            trainBERT = args.get("train_bert")
            humanBias = args.get("no_human_bias")
            verbose = args.get("verbose")
            
            if(verbose):
                os.environ.set("DEBUG","y")
            
            posApp = PosApp(
                datasetDir,
                tokenizer,
                lazyCache,
                minibatchSize,
                learningRate,
                threshold,
                patience,
                tEpochs,
                modelPath,
                update,
                trainBERT,
                humanBias
            )
            
                    
            log.error("Unimplemented")
            terminate(0)
        except KeyboardInterrupt as e:
            terminate(0)
        except Exception as e2:
            traceback = traceback.format_exc()
            log.error("Application stopped unexpectedly: "+str(traceback))
            terminate(1)

