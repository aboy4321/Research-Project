import matplotlib
from matplotlib import pyplot as plt
import itertools
import pygraphviz as pgv
import heapq

class ThresholdTest:
    id_counter = 0 # next available id
    def __init__(self, weights, threshold):
        #establishing variables
        self.weights = weights
        self.threshold = threshold
        self.size = len(weights)
        self.id = ThresholdTest.new_id()

    @classmethod
    def new_id(cls):
        new_id = cls.id_counter
        cls.id_counter += 1
        return new_id
    def __repr__(self):

        #printing threshold
        root = " + ".join([f"{weight}*x_{var+1}" for var, weight in enumerate(self.weights)])
        root = f"{root} >= {self.threshold}"
        if not self.weights:
            return f"0 >= {self.threshold}"
        return root

    def as_truth_table(self):
        passed = 0
        fail = 0
        pass_list = [0]
        fail_list = [0]

        all_combinations = list(itertools.product([0, 1], repeat=self.size))
        headers = [f"X_{i + 1}" for i in (range(self.size))]
        print("|".join(headers) + "|Result")

        #iterating through all possible combinations
        for c in all_combinations:
            #sums the weighted inputs from the combinations
            weighted_sum = sum(weight * inputs for weight, inputs in zip(self.weights, c))
            #satisfied variable returns True if test is passed and False otherwise
            satisfied = weighted_sum >= self.threshold
            if satisfied:
                passed += 1
                pass_list.append(passed)
                fail_list.append(fail)
            else:
                fail += 1
                fail_list.append(fail)
                pass_list.append(passed)
            #printing combinations seperated by lines for readability

            print(f"{c}, {satisfied}")
        return pass_list, fail_list

    #def

    def set_last_input(self, value):
        assert self.weights
        #when values in path are set to 1
        if value == 1:
            last_input = self.weights[-1]
            new_threshold = (self.threshold - last_input)
            new_weights = self.weights[:-1]
        #when values in path are set to 0
        elif value == 0:
            new_weights = self.weights[:-1]
            new_threshold = self.threshold
        return ThresholdTest(new_weights, new_threshold)

def print_path(test, values, depth=0):

    print("  " * depth + str(test))
    if not values:
        return

    next_value = values[-1]
    print("  " * depth + f"└-[x_{test.size}={next_value}]- ", end="")

    reduced_test = test.set_last_input(next_value)
    print_path(reduced_test, values[:-1], depth + 1)
    # ideally recursive
def print_tree(test, depth=0):
    print("  " * depth + str(test))
    if not test.weights:
        return

    next_value = 0
    reduced_test = test.set_last_input(next_value)
    print_tree(reduced_test, depth + 1)
    next_value = 1
    reduced_test = test.set_last_input(next_value)
    print_tree(reduced_test, depth + 1)

class TreePlotter():
    def __init__(self):
        #initializing a graph to use
        self.graph = pgv.AGraph(strict=True, directed=False)

    #function to add nodes:
    def add_node(self, node_id, label, color):
        self.graph.add_node(node_id, label=label, shape='box', color=color)

    #function to add edges
    def add_edge(self,  parent_id, child_id, label):
        self.graph.add_edge(parent_id, child_id, label=label)
        edge_count = 0
        edge_count += 1
    #saves the tree to a file using dot
    def draw_tree(self, filename="tree_plot.png"):
        self.graph.layout(prog="dot")
        self.graph.draw(filename)

def is_trivial_fail(test):
    bounds = Bounds(test.weights)
    upper = bounds.upper_bound()
    lower = bounds.lower_bound()
    threshold = test.threshold
    return lower <= upper < threshold

def is_trivial_pass(test):
    bounds = Bounds(test.weights)
    upper = bounds.upper_bound()
    lower = bounds.lower_bound()
    threshold = test.threshold
    return threshold <= lower <= upper

class Counter():
    def __init__(self,test_size):
        self.test_size = test_size
        self.passes = 0
        self.fails = 0
        self.pass_counts = [0]
        self.fail_counts = [0]

    def is_trivial_and_count(self,test):
        total_counts = 2 ** test.size
        if is_trivial_pass(test):
            self.passes += total_counts
            self.pass_counts.append(self.passes)
            self.fail_counts.append(self.fails)
            #print(f"Pass count: {self.pass_counts}")
            return True
        if is_trivial_fail(test):
            self.fails += total_counts
            self.fail_counts.append(self.fails)
            self.pass_counts.append(self.passes)
            #print(f"Fail count: {self.fail_counts}")
            return True
        return False
class Bounds():
    def __init__(self, weights):
        self.weights = weights

    def upper_bound(self):

        return sum(w for w in self.weights if w > 0)

    def lower_bound(self):
        return sum(w for w in self.weights if w <= 0)

    def gap_size(self, test):

        upper_diff = abs(self.upper_bound() - test.threshold)
        lower_diff = abs(test.threshold - self.lower_bound())

        score_list = [self.lower_bound(), lower_diff, test.threshold,upper_diff, self.upper_bound()]
        return score_list

def form_tree(plot, test, pass_steps, fail_steps, parent_id=None, depth=0, counter=None):

    pass_steps, fail_steps = steps_to_pass(test), steps_to_fail(test)

    bounds = Bounds(test.weights)

    count = 2 ** test.size

    current_label = f"{test}\n{bounds.gap_size(test)}\n{pass_steps}\n{fail_steps}\nCount: {count}"
    current_id = f"Node_{depth}_{test.id}"
    print(pass_steps, fail_steps)
    if is_trivial_pass(test):
        node_color = 'green'
    elif is_trivial_fail(test):
        node_color = 'red'
    else:
        node_color = 'black'
    plot.add_node(current_id, current_label, color = node_color)

    if parent_id is not None:
        plot.add_edge(parent_id, current_id, label=f"x_{test.size + 1}")

    if counter.is_trivial_and_count(test):
        return
    # Otherwise test is not a leaf
    for next_value in [1, 0]:

        reduced_test = test.set_last_input(next_value)
        form_tree(plot, reduced_test,pass_steps=pass_steps, fail_steps=fail_steps,parent_id=current_id, depth=depth + 1, counter=counter)
    plot.draw_tree("tree_plot.png")

def bfs_form_tree(plot, test, parent_id=None, depth=0):
    heap = []
    heapq.heappush(heap, (steps_to_pass(test), steps_to_fail(test), test))
    iteration = 0
    while heap:
        pass_steps, fail_steps, test = heapq.heappop(heap)

        bounds = Bounds(test.weights)

        count = 2 ** test.size

        current_label = f"{test}\n{bounds.gap_size(test)}\n{pass_steps}\n{fail_steps}\nCount: {count}, Iter. {iteration}"
        current_id = f"Node_{depth}_{test.id}"

        if is_trivial_pass(test):
            node_color = 'green'
        elif is_trivial_fail(test):
            node_color = 'red'
        else:
            node_color = 'black'
            left = test.set_last_input(0)
            heapq.heappush(heap, (steps_to_pass(left), steps_to_fail(left), left))
            
            right = test.set_last_input(1)
            heapq.heappush(heap, (steps_to_pass(right), steps_to_fail(right), right))
        plot.add_node(current_id, current_label, color = node_color)

        if parent_id is not None:
            plot.add_edge(parent_id, current_id, label=f"x_{test.size + 1}")

        if counter.is_trivial_and_count(test):
            continue
        
        
        iteration += 1
    plot.draw_tree("tree_plot.png")

def steps_to_pass(test):

    if is_trivial_pass(test):
        return -1

    for steps in range(1, test.size + 1):
        min_test = test
        for i in range(steps):
            if min_test.weights[-1] == 0:
                min_test = min_test.set_last_input(0)
            elif min_test.weights[-1] < 0:
                min_test = min_test.set_last_input(0)
            elif min_test.weights[-1] > 0:
                min_test = min_test.set_last_input(1)

        if is_trivial_pass(min_test):
            return steps
    return 0
def steps_to_fail(test):
    if is_trivial_fail(test):
        return -1

    for steps in range(1, test.size + 1):
        min_test = test
        for i in range(steps):
            if min_test.weights[-1] == 0:
                min_test = min_test.set_last_input(0)
            elif min_test.weights[-1] < 0:
                min_test = min_test.set_last_input(1)
            elif min_test.weights[-1] > 0:
                min_test = min_test.set_last_input(0)

        if is_trivial_fail(min_test):
            return steps

    return 0
# graph showing upper/lower bounds of the passes and fails of the threshold test
def pass_fail_graph(pass_list, fail_list, pruned_pass, pruned_fail, test):
    font_size = 20

    matplotlib.rcParams.update({'xtick.labelsize': font_size,
                                'ytick.labelsize': font_size,
                                'figure.autolayout': True})
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    # computing the top value (this is 2^n)
    total = 2 ** test.size
    # flipping the fail_list to start at the top and count down
    pruned_fail = [total- fail  for fail in pruned_fail]
    fail_list = [ total-fail for fail in fail_list ]
    # make sure pass_list and flipped fail_list meet at the same endpoint
    assert pass_list[-1] == fail_list[-1]
    # this is the final count
    count = pass_list[-1]

    plt.plot(pass_list,color='blue')
    plt.plot(fail_list,color='red')
    plt.plot(pruned_fail,color='red' , linestyle= '--')
    plt.plot(pruned_pass,color='blue' , linestyle= '--')
    plt.axhline(y=count, linestyle='--', color="purple")
    plt.show()

def score(test):
    #scored based on how close to trivial the node is
    pass

weights = [-1, 2, 4, 8, -11]

bounds = Bounds(weights)

threshold = 12
threshold_test = ThresholdTest(weights, threshold)

step1 =  steps_to_pass(threshold_test)
step2 =  steps_to_fail(threshold_test)
plotter = TreePlotter()
counter = Counter(threshold_test.size)
form_tree(plotter, threshold_test, step1,step2, counter=counter)
#bfs_form_tree(plotter, threshold_test, counter=counter)
#plotter.draw_tree("tree_plot.png")

pFail_list, pPass_list=  counter.fail_counts, counter.pass_counts
pass_list, fail_list = threshold_test.as_truth_table()

pass_fail_graph(pass_list, fail_list, pPass_list, pFail_list, threshold_test)
