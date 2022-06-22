from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

from numpy.core.function_base import linspace

num_of_points = 100

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/ 



def getPlot_line(wiggle=False):
    # Data for a three-dimensional line
    linspace_start=random.randint(0,50)
    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    a_size=random.randint(1,50)
    b_size=random.randint(1,50)

    x_dir=np.random.uniform(-1,1)
    y_dir=np.random.uniform(-1,1)

    zline = np.linspace(linspace_start, linspace_size, num_of_points)
    xline,yline=0,0
    if not wiggle:
        xline = (np.linspace(random.randint(0,50), random.randint(0,50), num_of_points)*x_dir)*x_size
        yline = (y_dir*a_size*np.linspace(random.randint(0,50), random.randint(0,50), num_of_points) + b_size*0.1)*y_size
    else:
        if random.choice([0,1])==0:
            xline = (np.linspace(random.randint(0,50), random.randint(0,50), num_of_points)*x_dir)*x_size
            yline = np.sin(zline)*y_size
        else:
            xline = np.sin(zline)*y_size
            yline = (y_dir*a_size*np.linspace(random.randint(0,50), random.randint(0,50), num_of_points) + b_size*0.1)*y_size
    # toCheck = [(yline[i],xline[i],zline[i]) for i in range(len(zline))]
    toReturn = [yline,xline,zline]

    return toReturn
    

def getPlot_line_wiggle():
    return getPlot_line(True)


def getPlot_parabole():
    ''' Get curved lines and paraboles
    '''
    #https://likegeeks.com/3d-plotting-in-python/

    x,y,z=0,0,0

    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    m,n = np.random.uniform(-1,1,[1,2])[0]

    x = np.linspace(-x_size*np.pi*np.random.uniform(0,1), x_size*np.pi*np.random.uniform(0,1),num_of_points)
    y = np.linspace(-y_size*np.pi*np.random.uniform(0,1), y_size*np.pi*np.random.uniform(0,1),num_of_points)
    z = m*x**2 + n*y**2

    toReturn = [y,x,z]

  
    return toReturn

def getPlot_spiral():
    # Data for a three-dimensional spiral
    linspace_size=random.randint(1,50)
    x_size=random.randint(1,50)
    y_size=random.randint(1,50)
    zline = np.linspace(0, linspace_size, num_of_points)
    xline = np.cos(zline)*x_size
    yline = np.sin(zline)*y_size
    # toCheck = [(yline[i],xline[i],zline[i]) for i in range(len(zline))]
    toReturn = [yline,xline,zline]

    return toReturn


func=[getPlot_line,getPlot_line_wiggle,getPlot_parabole,getPlot_spiral]
funcName=["line","wiggle","parabole","spiral"]

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
    

def getOrbit(how_many_to_return:int=1,f_type=None,numPoints=None):
    checkAmount(how_many_to_return)
    if numPoints != None:
        global num_of_points
        num_of_points=numPoints
    toReturn = []
    for i in range(how_many_to_return):
        if f_type != None:
            f_type=checkType(f_type)

        type_f = f_type
        if type_f == None:
            type_f = random.randint(0,len(func)-1)
        tup = (func[type_f]())  
        toReturn.append(tup)
    return toReturn



# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plt', action='store_true',help='to chart using matplotlib.pyplot')
    parser.add_argument('--type', action='store', default=random.randint(0,len(func)-1) ,help='the type of line to be generated')

    arguments = parser.parse_args()

    f_type=arguments.type

    yline, xline, zline,ydata,xdata,zdata=0,0,0,0,0,0

    ret = getOrbit(1,f_type)
    ret = ret[0]

    yline, xline, zline = ret[0:3]
    
    # yline, xline, zline = getPlot(random.randint(0, 3))
    if arguments.plt:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xline, yline, zline, 'green')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
        plt.show()
        pass