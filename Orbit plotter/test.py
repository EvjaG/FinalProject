import generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


yline, xline, zline,ydata,xdata,zdata=0,0,0,0,0,0
how_many_orbits = 1
ret1 = generator.getOrbit(how_many_orbits)[0]
# fig, ax = plt.subplots(1, how_many_orbits)
# fig.axes(projection='3d')
# for i,ret in enumerate(ret1):
#     yline, xline, zline = ret[0:3]
#     # ax[i] = plt.axes(projection='3d')
#     ax[i].plot3D(xline[0], yline[0], zline[0], 'blue')

#     ax[i].plot3D(xline, yline, zline, 'green')
#     ax[i].set_xlabel("x")
#     ax[i].set_ylabel("y")
#     ax[i].set_zlabel("z")

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',   '#7f7f7f', '#bcbd22', '#17becf']
def animate(num, data, line):
   line.set_color("#000000")
#    line.set_alpha(0.7)
   line.set_data(data[0:2, :num])
   line.set_3d_properties(data[2, :num])
   return line

x,y,t=ret1[0],ret1[1],ret1[3]
# data = np.array([x, y, t])
# N = len(t)
# fig = plt.figure(num="t=time")
# ax = Axes3D(fig)

# line, = plt.plot(data[0], data[1], data[2], lw=7, c='green')
# line_ani = animation.FuncAnimation(fig, animate, frames=N, fargs=(data, line), interval=50, blit=False)
# plt.show()

t=ret1[2]
data = np.array([x, y, t])
N = len(t)
fig = plt.figure(num="t=z")
ax = Axes3D(fig)

line, = plt.plot(data[0], data[1], data[2], lw=7, c='green')
line_ani = animation.FuncAnimation(fig, animate, frames=ret1[3], fargs=(data, line), interval=50, blit=False)


plt.show()


