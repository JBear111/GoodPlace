import sys
import os

import cv2

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    stringer = pathIn.split("/")
    season = stringer[1]
    epizoda = stringer[2].split(".")
    epizoda = epizoda[0]
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite(os.path.join(pathOut,"{}-{}-{}.jpg".format(season,epizoda,count)), image)     #format: S1-ep1-1.jpg
        count = count + 1   #every second saves a frame
        
if __name__=="__main__":
    filename = os.listdir(PATH)
    for file in filename:
        extractImages(SOURCE + file, DEST)
