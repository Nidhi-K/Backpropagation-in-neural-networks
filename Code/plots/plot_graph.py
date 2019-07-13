from matplotlib import pyplot
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='plot_graph.py')
parser.add_argument('--plots', nargs='+')
args = parser.parse_args()

num_plots = len(args.plots)
x = []; y = []
for i in np.arange(num_plots):
  y.append(np.load(args.plots[i]))
  x.append(np.arange(y[i].size))

for i in np.arange(num_plots):
  pyplot.plot(x[i], y[i], label = args.plots[i].replace('.npy', '')) 

pyplot.legend()
pyplot.show()
