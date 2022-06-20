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
csvWrite = True
# picWrite = True
picWrite = False
oneFolder = False

csv_loc = False # to include location in generated tables
csv_vel = True  # to include velocity in generated tables  
csv_acc = False # to include acceleration in generated tables
csv_tim = False # to include timestamp in generated tables

resizeFactor = 1
num_points=100
generator.num_of_points=num_points
# **************************************************
how_many_orbits = 20
num_of_orbit_types = len(generator.func)
funcNames = generator.funcName

def animateBFunc(plotA,unique_filename,saveGIF=saveGIF,showAnimation=showAnimation):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    def animate(num, data, line):
        line.set_color("#000000")
        #    line.set_alpha(0.7)
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line
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
    f=  open(pather+'.csv', 'w',newline='')
    # create the csv writer
    writer = csv.writer(f)
    # writer.writerow(["t","y","x","z","yv","xv","zv","ya","xa","za"])
    # writer.writerow(["t","x","y","z","vx","vy","vz"])



    data=plot
    time=0
    addTime = 0
    if(customTime):
      addTime = 1
    # write rows to the csv file
    for i in range(len(data[0])):
        t   = str(np.float16(time))
        if customTime:
            t = str(data[0][i])
        x   = str(data[0+addTime][i])
        y   = str(data[1+addTime][i])
        z   = str(data[2+addTime][i])
        yv  = str(data[3+addTime][i])
        xv  = str(data[4+addTime][i])
        zv  = str(data[5+addTime][i])
        ya  = str(data[6+addTime][i])
        xa  = str(data[7+addTime][i])
        za  = str(data[8+addTime][i])

        toAppend = []
        if csv_tim:
            toAppend.append(t)
        if csv_loc:
            toAppend.append(y)
            toAppend.append(x)
            toAppend.append(z)
        if csv_vel:
            toAppend.append(yv)
            toAppend.append(xv)
            toAppend.append(zv)
        if csv_acc:
            toAppend.append(ya)
            toAppend.append(xa)
            toAppend.append(za)
        
        writer.writerow(toAppend)
        time+=0.1

def picWriteFunc(plot,pather):
    arr = img = np.array(plot,dtype=np.float32)
    scale_percent = 100*resizeFactor # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    arr = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # arr = cv2.resize(arr,)
    # arr = np.vstack((arr,arr,arr))
    # arr = cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(pather+'.jpg',arr)


def mainFunc():
    #create data folder designations***************
    trainFunc = funcNames.copy()
    testFunc = funcNames.copy()
    for i in range(len(trainFunc)):
        trainFunc[i]    = f'data/train/{i}'
        testFunc[i]     = f'data/test/{i}'

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
        plot = generator.getOrbit(f_type=f_type)[0]
        if j >= 0.8 * how_many_orbits:
                traintest = "test"
        unique_filename = str(uuid.uuid4())
        f_type = funcNames[f_type]
        pather = f'./data/'
        
        if not oneFolder:
          pather += f"{traintest}/{funcNames.index(f_type)}/data_{unique_filename}"
        else:
          pather += unique_filename
        len_plot = len(plot[0])

        # normalize data
        #taking max and min
        my,ny = plot[0].max(),plot[0].min()
        mx,nx = plot[1].max(),plot[1].min()
        mz,nz = plot[2].max(),plot[2].min()
        #
        plot[0]=(plot[0]-ny)/(my-ny + 0.00001)
        plot[1]=(plot[1]-nx)/(mx-nx + 0.00001)
        plot[2]=(plot[2]-nz)/(mz-nz + 0.00001)


        # calculate velocities
        yv = (plot[0][1:len_plot] -plot[0][0:len_plot-1])/(0.1)
        xv = (plot[1][1:len_plot] -plot[1][0:len_plot-1])/(0.1)
        zv = (plot[2][1:len_plot] -plot[2][0:len_plot-1])/(0.1)
        yv=np.insert(yv,0,0)
        xv=np.insert(xv,0,0)
        zv=np.insert(zv,0,0)

        # calculate accelerations
        ya = (yv[1:len_plot] -yv[0:len_plot-1])/(0.1) 
        xa = (xv[1:len_plot] -xv[0:len_plot-1])/(0.1) 
        za = (zv[1:len_plot] -zv[0:len_plot-1])/(0.1) 
        ya=np.insert(ya,0,0)
        xa=np.insert(xa,0,0)
        za=np.insert(za,0,0)


        # attach velocities and accelerations to plot
        plot += [yv,xv,zv,ya,xa,za]

        if animateB:
            animateBFunc(plot[0:3],unique_filename)
        
        if csvWrite:
            csvWriteFunc(plot,pather)

        if picWrite:
            picWriteFunc(plot, pather)

if __name__ == '__main__':
    mainFunc()

