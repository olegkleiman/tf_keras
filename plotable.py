import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Plotable(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        plt.gcf().canvas.set_window_title('Neuron classification')

    def plot_input(self, samples, labels):
        for sample, label in zip(samples, labels):
            x, y = sample
            color, marker = ('green', '^') if label == 1 else ('red', 's')
            plt.scatter(x, y, color=color, marker=marker)

    def draw_separation_line(self, weights, error, loss):
        bias, w1, w2 = weights
        plt.plot([0, 5], 
          [-bias/w2, (-bias - w1*5)/w2]
        )
        
        _error = error if isinstance(error, float) else error.numpy()
        # _error = error.numpy()
        print(_error)
        plt.suptitle(f"w1={round(w1, 4)}, w2={round(w2, 4)}, bias={round(bias, 4)} error={round(_error, 4)}\n loss={loss}\ny={round((-w1)/w2, 4)}*x + {round(-bias/w2, 4)}", 
                     horizontalalignment = 'center',
                     fontsize='large')

    def redraw(self, inputs, labels, weights, error, loss):
        self.ax.clear()
        self.plot_input(inputs, labels)
        self.draw_separation_line(weights, error, loss) 

    def start(self, train_steps):
        self.anim = animation.FuncAnimation(self.fig, train_steps, interval=300)
        plt.show()