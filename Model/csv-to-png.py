import numpy as np
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join

def mn(frm:str,to:str):
    # print(data)
    # csv_reader = csv.reader(data, delimiter=',')
    onlyfiles = [f for f in listdir(frm) if isfile(join(frm, f))]
    for file in onlyfiles:
        data = np.genfromtxt(frm+'/'+file,delimiter=",")
        print(data)

    pass

mn("data/train/","")