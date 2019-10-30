import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import random as random
import keyboard

plt.switch_backend('macosx')

# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

labels = {
    "caterpilar": "1",
    "ladybird": "2"
}

# y > x
input_caterpilars = np.array([
    [1.,3.],
    [2.,3.],
    [0.,1.]
])

# x > y
input_ladybirds = np.array([
    [3.,1.],
    [2.,2.],
    [1,0.4]
])

def draw_input():
    x, y = input_caterpilars.T
    plt.scatter(x, y, color=['red'])
    x, y = input_ladybirds.T
    plt.scatter(x,y, color=['green'])

draw_input()

weight = 0.25 #random.random()
print(f"Initial weight: {weight}")

# draw separating line
def draw_separating_line(w):
    xl = ax.set_xlabel( f"weight: {w}")
    plane_point_x = input_ladybirds[0][0]
    plane_point_y = (input_ladybirds[0][1] + .1)
    plane_x = [0, plane_point_x]
    plane_y = [0, plane_point_x * w]
    ax.plot(plane_x, plane_y)

draw_separating_line(weight)

y = weight * input_ladybirds[0][0] # x
error = input_ladybirds[0][1] - y
print(f"Error: {error}")

current_training_set = input_ladybirds

def animate(i):
    global weight
    global current_training_set
    y = weight * current_training_set[0][0] # x
    error = (current_training_set[0][1] + .1) - y
    print(f"Iteration: {i} Error: {error}")
    if( error == .0 ):
        xl = ax.set_xlabel(f"Learned. Weight: {weight}")
        print(len(current_training_set))
        current_training_set = input_caterpilars
        # anim.event_source.stop()

    ax.clear()
    draw_input()
    delta = error / current_training_set[0][0]
    print(f"Iteration: {i} New ∆ weight: {delta}")    
    
    weight += delta  
    print(f"Iteration: {i} New weight: {weight}")
    draw_separating_line(weight)      

def init_animation():
    pass

anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=2000)
plt.show()
    




