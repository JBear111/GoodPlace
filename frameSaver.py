import sys
import argparse
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
        cv2.imwrite(os.path.join(pathOut,"{}-{}-{}.jpg".format(season,epizoda,count)), image)     #ime: ime npr. S1-ep1-1.jpg
        count = count + 1   #count + 3 za svake 3 sekunde...

if __name__=="__main__":
    filename = os.listdir("Torrent/S1")
    for file in filename:
        extractImages("Torrent/S1/" + file, "input/S1")

    filename = os.listdir("Torrent/S1")
    for file in filename:
        extractImages("Torrent/S1/" + file, "input/S1")

    filename = os.listdir("Torrent/S3")
    for file in filename:
        extractImages("Torrent/S3/"+ file, "input/S3")

    filename = os.listdir("Torrent/S4")
    for file in filename:
        extractImages("Torrent/S4/" + file, "input/S4")

#Torrent/S1/ep1.mkv
#input/S1