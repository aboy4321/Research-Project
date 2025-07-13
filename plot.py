from proto2 import Counter
import numpy as np
from matplotlib import pyplot as plt
import pickle

def plot_start():
    import matplotlib
    from matplotlib import pyplot as plt

    font_size = 20
    matplotlib.rcParams.update({'xtick.labelsize': font_size,
                                'ytick.labelsize': font_size,
                                'figure.autolayout': True})
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'pdf.use14corefonts' : True,
                                'text.usetex' : True})

def plot_one(counter, plot_times=True, linestyle='-'):

    pass_counts = counter.pass_counts
    fail_counts = counter.fail_counts

    # computing the top value (this is 2^n)
    total = 2 ** counter.test_size
    
    # Flipping the fail_list to start at the top and count down
    fail_counts = [total - count for count in fail_counts]
    
    # Makes sure pass_list and flipped fail_list meet at the same endpoint
    
    # Final count of the lists
    count = pass_counts[-1]
    if plot_times:
        times = counter.count_times[:len(pass_counts)]
    else:
        times = list(range(1,len(pass_counts)+1))
    plt.plot(times,pass_counts, color='blue', linestyle=linestyle)
    plt.plot(times,fail_counts, color='red', linestyle=linestyle)
    plt.axhline(y=count, linestyle='--', color="purple")

def plot_end():
    plt.xlabel('\\# of explanations')
    plt.ylabel('model count')

    plt.xscale("log")
    plt.savefig("plot-log.png")
    plt.xscale("linear")
    plt.savefig("plot-linear.png")
    plt.savefig("plot-linear.pdf")
    plt.show()


EXPERIMENT1 = True

if EXPERIMENT1:
    pickle_filename = "experiment1.pickle"

    #data = { 'counter_A': counter_A,
    #        'counter_R': counter_R }
    with open(pickle_filename,'rb') as f:
        data = pickle.load(f)

    plot_start()
    plot_one(data['counter_A'],plot_times=False,linestyle='-')
    plot_one(data['counter_R'],plot_times=False,linestyle=':')
    plot_end()
    exit()
