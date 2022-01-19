import generator
import numpy as np
import random
import os
import cv2
import uuid
import writer
import datetime

# https://xp-soaring.github.io/igc_file_format/index.html 



# What to do in iteration **************************
#animateB = False
animateB = False
showAnimation = False
saveGIF = False
csvWrite = False
#csvWrite = True
#picWrite = True
picWrite = True
# **************************************************

def transform(plot):
    Lines = plot.readlines()
    toReturn = []
    FirstLine = True
    last_time_delta=0
    last_time_sec=0
    last_alt=0
    for Line in Lines:
        if Line[0]!='B':
            continue

        #calculate the time diff from last line
        Line=Line.strip()[1:-5]
        time = Line[0:6]
        time = datetime.datetime.strptime(time,'%H%M%S')
        td = time = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
        if FirstLine:
            last_time_delta   = time
            time        = 0
            FirstLine   = False
        else:
            delta       = (time - last_time_delta).total_seconds()
            time        = last_time_sec+delta
            last_time_sec   = time
            last_time_delta = td
            pass

        # find the lon,lat,alt
        Line=Line[6:]

        # lat
        latIDX = Line.find('N')
        if latIDX<0:
            latIDX = Line.find('S')
        lat = Line[0:latIDX]

        Line = Line[latIDX+1:]
        #lon
        lonIDX = Line.find('W')
        if lonIDX<0:
            lonIDX = Line.find('E')
        lon = Line[0:lonIDX]
        Line = Line[lonIDX+1:]
        
        alt=last_alt
        if Line[0]=='A':
            alt=Line[1:7]
        
        toReturn.append([time,lat,lon,alt])

    return toReturn


if __name__ == '__main__':
    data_folder_location = 'igc'
    if not os.path.exists(data_folder_location):
        exit(1)

    # load images from folder
    data_folder=[f for f in sorted(os.listdir(data_folder_location)) if (str(f))[-3:] == "igc"]
    if len(data_folder) == 0:
        exit(1)
    folders=['data','data/train/igc','data/test/igc']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    del os
    igc_files = []
    for file in data_folder:
        f=open(f'{data_folder_location}/{file}','r')
        igc_files.append((f,file[:-4]))
    del data_folder
    writable_files = []
    for file in igc_files:
        transformed_path = transform(file[0])
        file[0].close()
        transformed_path=np.array(transformed_path).T.astype(np.float32)
        writable_files.append((transformed_path,file[1]))
    for file in writable_files:
        if csvWrite:
            writer.csvWriteFunc(file,f'./data/train/igc/{file[1]}',True)
        if showAnimation or saveGIF:
            writer.animateBFunc([file[0][1:]],file[1],saveGIF,showAnimation)
        if picWrite:
            writer.picWriteFunc(file[0][1:],f'data/train/igc/{file[1]}')