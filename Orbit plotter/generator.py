from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

from numpy.core.function_base import linspace


# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/ 



def getPlot_line(points=False,wiggle=False):
    # Data for a three-dimensional line
    linspace_start=random.randint(0,50)
    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    a_size=random.randint(1,50)
    b_size=random.randint(1,50)

    x_dir=np.random.uniform(-1,1)
    y_dir=np.random.uniform(-1,1)

    zline = np.linspace(linspace_start, linspace_size, 100)
    xline,yline=0,0
    if not wiggle:
        xline = (np.linspace(random.randint(0,50), random.randint(0,50), 100)*x_dir)*x_size
        yline = (y_dir*a_size*np.linspace(random.randint(0,50), random.randint(0,50), 100) + b_size*0.1)*y_size
    else:
        if random.choice([0,1])==0:
            xline = (np.linspace(random.randint(0,50), random.randint(0,50), 100)*x_dir)*x_size
            yline = np.sin(zline)*y_size
        else:
            xline = np.sin(zline)*y_size
            yline = (y_dir*a_size*np.linspace(random.randint(0,50), random.randint(0,50), 100) + b_size*0.1)*y_size
    # toCheck = [(yline[i],xline[i],zline[i]) for i in range(len(zline))]
    toReturn = [yline,xline,zline]

    if points:
        # Data for three-dimensional scattered points
        zdata = linspace_size * np.random.random(100)
        xdata = (np.sin(zdata) + 0.1 * np.random.randn(100))*x_size
        ydata = (np.cos(zdata) + 0.1 * np.random.randn(100))*y_size
        toCheck_points = [(ydata[i],xdata[i],zdata[i]) for i in range(len(zdata))]
        toReturn = (toReturn,[ydata,xdata,zdata])
    return toReturn
    

def getPlot_line_wiggle(points=False):
    return getPlot_line(points,True)


def getPlot_parabole(points=False):
    ''' Get curved lines and paraboles
    '''
    #https://likegeeks.com/3d-plotting-in-python/

    x,y,z=0,0,0

    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    m,n = np.random.uniform(-1,1,[1,2])[0]

    x = np.linspace(-x_size*np.pi*np.random.uniform(0,1), x_size*np.pi*np.random.uniform(0,1),100)
    y = np.linspace(-y_size*np.pi*np.random.uniform(0,1), y_size*np.pi*np.random.uniform(0,1),100)
    z = m*x**2 + n*y**2

    toReturn = [y,x,z]

    if points:
        # Data for three-dimensional scattered points
        xdata = (x + 1.1 * np.random.randn(100))
        ydata = (y + 1.1 * np.random.randn(100))
        # zdata = z
        zdata = m*xdata**2 + n*ydata**2
        toCheck_points = [(ydata[i],xdata[i],zdata[i]) for i in range(len(zdata))]
        toReturn = (toReturn,[ydata,xdata,zdata])

    return toReturn

def getPlot_spiral(points=False):
    # Data for a three-dimensional spiral
    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    zline = np.linspace(0, linspace_size, 100)
    xline = np.cos(zline)*x_size
    yline = np.sin(zline)*y_size
    # toCheck = [(yline[i],xline[i],zline[i]) for i in range(len(zline))]
    toReturn = [yline,xline,zline]

    if points:
        # Data for three-dimensional scattered points
        zdata = linspace_size * np.random.random(100)
        xdata = (np.sin(zdata) + 0.1 * np.random.randn(100))*x_size
        ydata = (np.cos(zdata) + 0.1 * np.random.randn(100))*y_size
        toCheck_points = [(ydata[i],xdata[i],zdata[i]) for i in range(len(zdata))]
        toReturn = (toReturn,[ydata,xdata,zdata])
    return toReturn


func=[getPlot_line,getPlot_line_wiggle,getPlot_parabole,getPlot_spiral]

def checkType(f_type):
    try:
        f_type=int(f_type)
        if f_type < 0 or f_type >= len(func):
            raise argparse.ArgumentError
    except:
        print("Can't interpret type, please only use integers from 0 to ",len(func),f"(not including {len(func)})")
        exit(1)
    return f_type
def checkAmount(num):
    if num <= 0:
        raise argparse.ArgumentError(None,"Argument invalid, amount to store should be 1 or more")
    
def timeFunc(zline):
    size = len(zline)
    toReturn = np.zeros(size)
    for i in range(len(zline)-1):
        size-=1
        toAdd = np.random.uniform(low=0.1, high=1, size=(size))
        toReturn[-size:] += toAdd
    toReturn = np.array(sorted(toReturn)).astype(np.float32)
    check = (toReturn[1:]-toReturn[0:-1])
    return toReturn

def getOrbit(how_many_to_return:int,f_type=random.randint(0,len(func)-1),points=False):
    f_type=checkType(f_type)
    checkAmount(how_many_to_return)
    toReturn = []
    for i in range(how_many_to_return):
        tup = (func[f_type](points))
        if not points:
            tup.append(timeFunc(tup[2]))
        else:
            tup[0].append(timeFunc(tup[0][2]))    
        toReturn.append(tup)
    return toReturn



# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', action='store_true',help='to also chart points around the line')
    parser.add_argument('--plt', action='store_true',help='to chart using matplotlib.pyplot')
    parser.add_argument('--type', action='store', default=random.randint(0,len(func)) ,help='the type of line to be generated')

    arguments = parser.parse_args()

    points,f_type=arguments.points,arguments.type

    yline, xline, zline,ydata,xdata,zdata=0,0,0,0,0,0

    ret = getOrbit(1,f_type,points)
    ret = ret[0]

    if not points:
        yline, xline, zline = ret[0:3]
    else:
        yline, xline, zline = ret[0][0:3]
        ydata,xdata,zdata   = ret[1]
    # yline, xline, zline = getPlot(random.randint(0, 3))
    if arguments.plt:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xline, yline, zline, 'green')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if points:
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
        plt.show()
        pass