import generator
import numpy as np
import random
import os
import cv2
import uuid
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


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
how_many_orbits = 500
num_of_orbit_types = len(generator.func)
funcNames = generator.funcName

def animateBFunc(plot,unique_filename,saveGIF=saveGIF,showAnimation=showAnimation):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    def animate(num, data, line):
        line.set_color("#000000")
        #    line.set_alpha(0.7)
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line
    plotA = plot[0]
    x,y,z=plotA[0],plotA[1],plotA[2]

    data = np.array([x, y, z])
    fig = plt.figure(num="t=z")
    ax = Axes3D(fig)

    line, = plt.plot((data[0]),(data[1]),(data[2]), lw=5,ls="-", c='green')
    line_ani = animation.FuncAnimation(fig, animate, frames=len(plotA[0]), fargs=(data, line), interval=1000/60, blit=False)
    if showAnimation:
        plt.show(block=True)
    if saveGIF:
        line_ani.save(f"./data/gifs/{unique_filename}.gif", dpi=300, writer=animation.PillowWriter(fps=40))


def csvWriteFunc(plot,pather,customTime=False):
    f=  open(pather+'.csv', 'w+',newline='')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["t","x","y","z"])
    # writer.writerow(["t","x","y","z","vx","vy","vz"])

    data=plot[0]
    time=0
    # write rows to the csv file
    if customTime:
        for i in range(len(data[0])):
            t = (str(data[0][i]))
            x = (str(data[1][i]))
            y = (str(data[2][i]))
            z = (str(data[3][i]))
            writer.writerow([t,x,y,z])
    else:
        for i in range(len(data[0])):
            t = (str(np.float16(time)))
            x = (str(data[0][i]))
            y = (str(data[1][i]))
            z = (str(data[2][i]))
            writer.writerow([t,x,y,z])
            time+=0.1

def picWriteFunc(plot,pather):
    arr = img = np.array(plot,dtype=np.float32)
    scale_percent = 1000 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    arr = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # arr = cv2.resize(arr,)
    # arr = np.vstack((arr,arr,arr))
    # arr = cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(pather+'.jpg',arr)


if __name__ == '__main__':
    #create data folder designations***************
    trainFunc = funcNames.copy()
    testFunc = funcNames.copy()
    for i in range(len(trainFunc)):
        trainFunc[i]    = 'data/train/'+trainFunc[i]
        testFunc[i]     = 'data/test/'+testFunc[i]

    folders = ['data','data/gifs','data/train','data/test'] + trainFunc + testFunc
    #check and create data folders if not created
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    #**********************************************

    f=[]
    writer = []

    traintest = "train"

    for j in range(how_many_orbits):
        f_type = random.randint(0, num_of_orbit_types-1)
        plot = generator.getOrbit(f_type=f_type)
        if j >= 0.8 * how_many_orbits:
                traintest = "test"
        unique_filename = str(uuid.uuid4())
        f_type = funcNames[f_type]
        pather = f'./data/{traintest}/{f_type}/data_{unique_filename}'

        if animateB:
            animateBFunc(plot,unique_filename)

        
        
        
        if csvWrite:
            csvWriteFunc(plot,pather)

        if picWrite:
            picWriteFunc(plot[0], pather)
