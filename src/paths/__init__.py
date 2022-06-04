import os
import sys

USERHOME = os.path.expanduser("~")
XDG_DATA_HOME = os.getenv("XDG_DATA_HOME")
if(not XDG_DATA_HOME):
    XDG_DATA_HOME = os.path.join(USERHOME,".local","share")
PROJECTNAME = os.path.basename(sys.argv[0]).replace(".py","")

APPDATA_DIR = os.path.join(XDG_DATA_HOME,PROJECTNAME)
APPDATA_SCRAPER = os.path.join(APPDATA_DIR,"traindata")
APPDATA_MODELS = os.path.join(APPDATA_DIR,"models")

os.makedirs(APPDATA_DIR,exist_ok=True)
os.makedirs(APPDATA_SCRAPER,exist_ok=True)
os.makedirs(APPDATA_MODELS,exist_ok=True)