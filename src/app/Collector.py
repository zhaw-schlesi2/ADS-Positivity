import logging
import requests
import os
import json
from src import paths
import traceback
import re
from src.app.ReviewDataSet import ReviewDataSet

class Collector():
    '''
    Searches and collects reviews from IMDB
    '''
    def __init__(self,apikey,targetdir=paths.APPDATA_SCRAPER,permissive=False,minRating=ReviewDataSet.minRating,maxRating=ReviewDataSet.maxRating):
        '''
        apikey - The api key from your imdb account
        targetdir - destination directory for the reviews. Defaults to XDG_DATA_HOME/.local/share/positivity/traindata
        permissive - if false only pick the first best match, if true collect reviews from all matches.
        minRating - Minimum rating the reviews must have, otherwise they are rejected
        maxRating - Maxmimum rating the reviews can have, otherwise they are rejected
        '''
        self.permissive = permissive
        self.targetdir=targetdir
        self.apikey=apikey
        self.language="de"
        self.apiSearchBase = f"https://imdb-api.com/{self.language}/API/SearchAll"
        self.apiReviewsBase = f"https://imdb-api.com/{self.language}/API/Reviews"
        self.filterMinRating = minRating
        self.filterMaxRating = maxRating
        self.commonHeaders = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        self.log = logging.getLogger(__name__+"."+__class__.__name__)
        
    def searchAndFetch(self,titles):
        '''
        titles - list with movie titles to search and fetch reviews for
        '''
        ids = self.searchMovies(titles)
        self.fetchReviews(ids)
        
    def searchMovies(self,titles):
        '''
        titles - search movies by a list of titles
        '''
        self.log.info("Searching IMDB for titles...")
        s = requests.Session()
        s.headers.update(self.commonHeaders)
        foundT = 0
        ids = []
        for title in titles:
            url = f"{self.apiSearchBase}/{self.apikey}/{title}"
            rsp = s.get(url,headers=self.commonHeaders)
            scode = rsp.status_code
            if(scode == 404):
                self.log.warn(f"Title [{title}] not found on IMDB")
                continue
            if(scode != 200):
                self.log.warn(f"Response code [{scode}] for request: [{url}] - skip")
                continue
            jsonData = rsp.json()
            try:
                results = jsonData.get("results")
                if(not results):
                    why = jsonData.get("errorMessage")
                    if(not why):
                        why = ""
                    self.log.warn(f"No search results found for [{title}] {why}")
                    continue
                found = 0
                for result in results:
                    if(found >=1 and not self.permissive):
                        self.log.info(f"[{len(results)}] Results found for [{title}] - picking first")
                        break
                    imdbID = result.get("id")
                    if(imdbID.startswith("tt")):
                        found+=1
                        foundT+=1
                        ids.append(imdbID)
                if(foundT % 10 == 0):
                    self.log.info(f"Looked up ID's of {foundT}/{len(titles)} titles...")
            except Exception:
                self.log.warn(f"Skipping request {url} due to exception")
                print(traceback.format_exc())
            finally:
                try:
                    rsp.close()
                except:
                    self.log.warn("Connection was probably already closed")
                    print(traceback.format_exc())
        self.log.info(f"Found {foundT} IDs in total for the given movie titles!")
        return ids
    
    
    def fetchReviews(self,ids):
        '''
        ids - a list of imdb movie ids to fetch reviews from
        '''
        self.log.info("Fetching reviews...")
        s = requests.Session()
        s.headers.update(self.commonHeaders)
        
        for id in ids:
            url = f"{self.apiReviewsBase}/{self.apikey}/{id}"
            rsp = s.get(url,headers=self.commonHeaders)
            scode = rsp.status_code
            if(scode != 200):
                self.log.warn(f"Response code {scode} for request: {url} - skip")
                continue
            jsonData = rsp.json()
            try:
                results = jsonData.get("items")
                if(not results):
                    self.log.warn(f"No results found for IMDB ID [{id}]")
                    continue
    
                tRev = len(results)
                tAccepted = 0
                tRejected = 0
                
                tfile = os.path.join(self.targetdir,str(id)+".jsonl")
                if(os.path.exists(tfile)):
                    self.log.warn(f"Overriding reviews for {id} with more recent data")
                    os.remove(tfile)
                tID = id
                with open(tfile,"a+") as fw:
                    for review in results:
                        text = review.get("content")
                        if(len(text.split(" ")) <= 10):#don't store reviews with less than aprx 10 words.
                            tRejected+=1
                            continue

                        #has the rate attribute
                        rate = review.get("rate")
                        if(not rate or len(rate) == 0):
                            tRejected+=1
                            continue
                        
                        #matches filters
                        if(float(rate) < self.filterMinRating or float(rate) > self.filterMaxRating):
                            tRejected+=1
                            continue
                        
                        #has the helpful attribute - if so extract
                        helpful = review.get("helpful")
                        if(not helpful or len(helpful) == 0):
                            tRejected+=1
                            continue
                        
                        helpful = helpful.replace(",",".")
                        reHelpful = re.search('([0-9\.]+).*?([0-9.]+)',helpful)
                        if(not len(reHelpful.groups()) == 2):
                            tRejected+=1
                            continue
                        
                        positive = float(reHelpful.group(1))
                        negative = float(reHelpful.group(2))
                        reviewHelpfulness = 0.0
                        if(positive+negative != 0.0):
                            reviewHelpfulness=positive/(positive+negative)#percent
                        jsonStr = json.dumps({
                            "rating":float(rate),
                            "helpful":reviewHelpfulness,
                            "text":text
                        })
                        fw.write(jsonStr+"\n")
                        fw.flush()
                        tAccepted+=1
                self.log.info(f"[{tID}]: Stored [{tAccepted}] reviews and filtered [{tRejected}] reviews out of [{tRev}]")
            except Exception:
                self.log.warn(f"Skipping request {url} due to exception")
                print(traceback.format_exc())
            finally:
                try:
                    rsp.close()
                except Exception:
                    self.log.warn("Connection was probably already closed")
                    print(traceback.format_exc())
            
            
            
