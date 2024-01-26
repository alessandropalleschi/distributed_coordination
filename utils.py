import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
class RunningStats:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.variance = np.zeros(shape)
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=(0, 2), keepdims=True)
        batch_variance = np.var(x, axis=(0, 2), keepdims=True)
        batch_count = x.shape[0]

        new_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean = self.mean + delta * batch_count / new_count
        if new_count == 1:
            self.variance = batch_variance
        else:
            self.variance = (
                self.variance * (self.count - 1) + batch_variance * (batch_count - 1) +
                delta**2 * self.count * batch_count / new_count
            ) / (new_count - 1)

        self.count = new_count

    def get_stats(self):
        return self.mean, np.sqrt(self.variance)