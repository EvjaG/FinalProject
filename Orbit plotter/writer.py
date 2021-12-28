from numpy.core.shape_base import block
import generator
import numpy as np


how_many_orbits = 3
plot_data = generator.getOrbit(how_many_orbits)

animateB = False
# animateB = True
# csvWrite = False
csvWrite = True

if animateB:

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    def animate(num, data, line):
        line.set_color("#000000")
        #    line.set_alpha(0.7)
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line
f=[]
writer = []




for plot in plot_data:
    if animateB:
        x,y,z=plot[0],plot[1],plot[2]

        data = np.array([x, y, z])
        fig = plt.figure(num="t=z")
        ax = Axes3D(fig)

        line, = plt.plot((data[0]),(data[1]),(data[2]), lw=5,ls="-", c='green')
        line_ani = animation.FuncAnimation(fig, animate, frames=len(plot[0]), fargs=(data, line), interval=1000/60, blit=False)
        plt.show(block=True)

    if csvWrite:
        import csv
        import uuid
        unique_filename = str(uuid.uuid4())

        f=  open(f'./data_{unique_filename}.csv', 'w+',newline='')
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(["t","x","y","z","vx","vy","vz"])

        data=plot
        time=0
        # write rows to the csv file
        for i in range(len(data[0])):
            t = (str(np.float16(time)))
            x = (str(data[0][i]))
            y = (str(data[1][i]))
            z = (str(data[2][i]))
            writer.writerow([t,x,y,z])
            time+=0.1
        
    pass