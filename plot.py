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
    
    if plot_times:
        times = counter.count_times[:len(pass_counts)]
        times = np.array(times) + 0.001 # HACK to get semilogx to work
    else:
        times = list(range(1,len(pass_counts)+1))
    plt.plot(times,pass_counts, color='blue', linestyle=linestyle)
    plt.plot(times,fail_counts, color='red', linestyle=linestyle)

    if pass_counts[-1] == fail_counts[-1]:
        # Final count of the lists
        count = pass_counts[-1]
        plt.axhline(y=count, linestyle='--', color="purple")

def plot_end(plot_times=True):
    if plot_times:
        plt.xlabel('time ($s$)')
    else:
        plt.xlabel('\\# of explanations')
    plt.ylabel('model count')
    
    plt.xscale("log")
    plt.savefig("plot-log.png")
    plt.xscale("linear")
    plt.savefig("plot-linear.png")
    plt.savefig("plot-linear.pdf")
    plt.show()


EXPERIMENT1 = False
EXPERIMENT2 = True
EXPERIMENT3 = False

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

if EXPERIMENT2:
    pickle_filename = "experiment2.pickle"
    timeout = 60

    with open(pickle_filename,'rb') as f:
        data = pickle.load(f)

    """
    tree_times = [ data[(i,j,mode)] for i,j,mode in data if mode == 'tree' ]
    graph_times = [ data[(i,j,mode)] for i,j,mode in data if mode == 'graph' ]

    not_finished = sum( 1 for time in tree_times if time >= timeout )
    finished = sum( 1 for time in tree_times if time < timeout )
    fin_times = [ time for time in tree_times if time < timeout ]
    print(f"tree: {finished}/{len(tree_times)} finished, avg. time: {np.mean(fin_times)}")

    not_finished = sum( 1 for time in graph_times if time >= timeout )
    finished = sum( 1 for time in graph_times if time < timeout )
    fin_times = [ time for time in graph_times if time < timeout ]
    print(f"graph: {finished}/{len(graph_times)} finished, avg. time: {np.mean(fin_times)}")
    """

    graph_counter = data['graph']
    tree_counter = data['tree']

    plot_start()
    plot_one(tree_counter,plot_times=True,linestyle=':')
    plot_one(graph_counter,plot_times=True,linestyle='-')
    plot_end(plot_times=True)

if EXPERIMENT3:
    pickle_filename = "experiment3.pickle"
    timeout = 60

    with open(pickle_filename,'rb') as f:
        data = pickle.load(f)
    locals().update(data)

    plot_start()
    plt.plot(tree_ns,tree_times,color='black',linestyle=':')
    plt.plot(graph_ns,graph_times,color='black',linestyle='-')
    plt.xlabel('$n$')
    plt.ylabel('time ($s$)')
    plt.xscale("log")

    axis = list(plt.axis())
    axis[2] = -5
    axis[3] = 65
    plt.axis(axis)
    #plt.show()

    plt.savefig("plot-times.png")
    plt.savefig("plot-times.pdf")
