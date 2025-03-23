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
    def __lt__(self,other):
        return self.id < other.id
    
    def __repr__(self):

        #printing threshold
        # Creates a representation of the threshold test inside of the tree
        root = " + ".join([f"{weight}*x_{var+1}" for var, weight in enumerate(self.weights)])
        root = f"{root} >= {self.threshold}"
        if not self.weights:
            return f"0 >= {self.threshold}"
        return root
    
    # Form_tree (pruned tree) is based off this truth table
    def as_truth_table(self):
        passed = 0
        fail = 0
        pass_list = [0]
        fail_list = [0]

        all_combinations = list(itertools.product([0, 1], repeat=self.size))
        headers = [f"X_{i + 1}" for i in (range(self.size))]
        #print("|".join(headers) + "|Result")

        #iterating through all possible combinations
        for c in all_combinations:
            #sums the weighted inputs from the combinations
            weighted_sum = sum(weight * inputs for weight, inputs in zip(self.weights, c))
            #satisfied variable returns True if test is passed and False otherwise
            satisfied = weighted_sum >= self.threshold

            # With this passes increase while fails remain stagnant, and vice versa
            if satisfied:
                passed += 1
                pass_list.append(passed)
                fail_list.append(fail)
            else:
                fail += 1
                fail_list.append(fail)
                pass_list.append(passed)
        
        # Returns list of passes and fails from the truth table
        return pass_list, fail_list
    
    # Function used to compute the threshold test
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

# Functions previously used to display the computations the threshold test was making
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

# Class used for the making of the decision tree, utilizes pygraphviz
class TreePlotter():
    def __init__(self):
        # Initializing a graph, in this case being a strict one
        self.graph = pgv.AGraph(strict=True, directed=False)

    # Function used to add nodes to the tree:
    def add_node(self, node_id, label, color):
        self.graph.add_node(node_id, label=label, shape='box', color=color)

    # Function to add edges to the nodes, from parent to child
    def add_edge(self,  parent_id, child_id, label):
        self.graph.add_edge(parent_id, child_id, label=label)
       # edge_count = 0
       # edge_count += 1
    # Draws the actual tree and saves the image as a .png file
    def draw_tree(self, filename="tree_plot.png"):
        self.graph.layout(prog="dot")
        self.graph.draw(filename)

# Functions testing for triviality:

def is_trivial_fail(test):
    bounds = Bounds(test.weights)
    upper = bounds.upper_bound()
    lower = bounds.lower_bound()
    threshold = test.threshold
    # True if the lower and upper are both less than the threshold
    return lower <= upper < threshold

def is_trivial_pass(test):
    bounds = Bounds(test.weights)
    upper = bounds.upper_bound()
    lower = bounds.lower_bound()
    threshold = test.threshold
    # True if both the lower and upper are greater than the threshold
    return threshold <= lower <= upper

# Class used to create an array of pass and fails 
class Counter():
    def __init__(self,test_size):
        # initializing variables
        self.test_size = test_size
        self.passes = 0
        self.fails = 0
        self.pass_counts = [0]
        self.fail_counts = [0]
    
    # Using previous functions of triviality, counts passes and fails
    def is_trivial_and_count(self,test):
        total_counts = 2 ** test.size
        if is_trivial_pass(test):
            self.passes += total_counts
            self.pass_counts.append(self.passes)
            self.fail_counts.append(self.fails)
            return True
        if is_trivial_fail(test):
            self.fails += total_counts
            self.fail_counts.append(self.fails)
            self.pass_counts.append(self.passes)
            return True
        return False

# Class used to measure the upper and lower bounds of the threshold test    
class Bounds():
    def __init__(self, weights):
        self.weights = weights

    def upper_bound(self):
        return sum(w for w in self.weights if w > 0)

    def lower_bound(self):
        return sum(w for w in self.weights if w <= 0)

    # Measures the differences of the upper and lower bound from the respective threshold
    def gap_size(self, test):

        upper_diff = abs(self.upper_bound() - test.threshold)
        lower_diff = abs(test.threshold - self.lower_bound())

        score_list = [self.lower_bound(), lower_diff, test.threshold,upper_diff, self.upper_bound()]
        return score_list

# Function used to create the pruned tree using a depth-first search algorithm
def form_tree(plot, test, pass_steps, fail_steps, parent_id=None, depth=0, counter=None):
    
    # Displaying important values onto the nodes (mainly steps to pass and fail)
    pass_steps, fail_steps = steps_to_pass(test), steps_to_fail(test)

    bounds = Bounds(test.weights)

    count = 2 ** test.size

    current_label = f"{test}\n{bounds.gap_size(test)}\n{pass_steps}\n{fail_steps}\nCount: {count}"
    current_id = f"Node_{depth}_{test.id}"
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

    # Iterates between the binary inputs of 0 and 1
    for next_value in [1, 0]:
       
        # Recursion, adds upon the depth of the tree
        reduced_test = test.set_last_input(next_value)
        form_tree(plot, reduced_test,pass_steps=pass_steps, fail_steps=fail_steps,parent_id=current_id, depth=depth + 1, counter=counter)
    
    plot.draw_tree("tree_plot.png")

# Improved form_tree function using a best-first search algorithm
# with the use of heaps
# NOTE: Will be going through this function step-by-step
def bfs_form_tree(plot, test, parent_id=None, depth=0, counter=None):
    # Initializing an empty heap array we will be iterating upon:
    heap = []
    initial_priority = min(steps_to_pass(test), steps_to_fail(test))
    # Creating and "pushing" our priorities to the heap
    heapq.heappush(heap, (initial_priority, test, parent_id, depth))
    iteration = 0

    # "while" The heap has values within itself
    while heap:
        
        # Removes the smallest of these variables from the heap and returns it
        priority, test, parent_id, depth = heapq.heappop(heap)

        current_priority = min(steps_to_pass(test), steps_to_fail(test))   
        # Information added onto the nodes within the tree
        bounds = Bounds(test.weights)

        count = 2 ** test.size

        current_label = f"{test}\n{bounds.gap_size(test)}\nCount: {count}, Iter. {iteration}"
        current_id = f"Node_{depth}_{test.id}"
        
        # Setting colors of the nodes based off passing or failing
        if is_trivial_pass(test): 
            node_color = 'green'
            
        elif is_trivial_fail(test):       
            node_color = 'red'
        
        # The bread and butter of this function. Using how close a test is to triviality, takes priority upon
        # the node that is closer (in this case being closer to passing) and pushes that value onto the heap
        else:
            node_color = 'black'
            
            left = test.set_last_input(0)
            right = test.set_last_input(1)

            left_priority = min(steps_to_pass(left), steps_to_fail(left))
            right_priority = min(steps_to_pass(right), steps_to_fail(right))

            heapq.heappush(heap, (left_priority, left, current_id, depth+1))
            heapq.heappush(heap, (right_priority, right, current_id, depth+1))

        # Adding nodes to tree
        plot.add_node(current_id, current_label, color=node_color)
        
        # While there are still parents within the tree, add edges
        if parent_id is not None:
            plot.add_edge(parent_id, current_id, label=f"x_{test.size + 1}")
        
        # When the test/node is trivial, continue and don't make computations upon the already trivial test
        if counter.is_trivial_and_count(test):
            continue
        
        # A count just to check for effiency :)
        iteration += 1
    
    # Draws the tree
    plot.draw_tree("tree_plot.png")

# ^^Function continues until heap is empty^^

# Iterates throught the amount of steps a node is away from becoming trivial
# using the set_last_input to check
def steps_to_pass(test):
    
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
    
    for steps in range(1, test.size + 1):
        min_test = test
        for _ in range(steps):
            if min_test.weights[-1] == 0:
                min_test = min_test.set_last_input(0)
            elif min_test.weights[-1] < 0:
                min_test = min_test.set_last_input(1)
            elif min_test.weights[-1] > 0:
                min_test = min_test.set_last_input(0)

        if is_trivial_fail(min_test):
            return steps

    return 0

# Graph showing the upper and lower bounds of the threshold test, and the steps taken to reach that number
# Mainly going to be used to show the efficiency of our algorithm

# The BFS, DFS and raw values are taken as parameters
def pass_fail_graph(bfs_pass, bfs_fail, pass_list, fail_list, pruned_pass, pruned_fail, test):
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
    
    # Flipping the fail_list to start at the top and count down
    pruned_fail = [total - fail for fail in pruned_fail]
    fail_list = [total - fail for fail in fail_list]
    bfs_fail = [total - fail for fail  in bfs_fail]
    
    # Makes sure pass_list and flipped fail_list meet at the same endpoint
    assert pass_list[-1] == fail_list[-1]
    
    # Final count of the lists
    count = pass_list[-1]
   
    plt.plot(pass_list,color='blue')
    plt.plot(fail_list,color='red')
    plt.plot(pruned_fail,color='red' , linestyle= '--')
    plt.plot(pruned_pass,color='blue' , linestyle= '--')
    plt.plot(bfs_pass, color='purple', linestyle='--')
    plt.plot(bfs_fail,color='purple', linestyle='--') 
    plt.axhline(y=count, linestyle='--', color="purple")

    plt.show()


#weights = [-1, 2, 4, 8, -11]
#weights = [-2, 3, -4, 5]
#weights = [-30, 4, 8, 22, 9, 12, -17]
weights = [1, -1, 2, -2, 4,-4, 8 -8]
bounds = Bounds(weights)

threshold = 1
threshold_test = ThresholdTest(weights, threshold)

step1 =  steps_to_pass(threshold_test)
step2 =  steps_to_fail(threshold_test)
plotter = TreePlotter()
counter = Counter(threshold_test.size)
bfs_counter = Counter(threshold_test.size)
form_tree(plotter, threshold_test, step1,step2, counter=counter)

pFail_list, pPass_list = counter.fail_counts, counter.pass_counts
pass_list, fail_list = threshold_test.as_truth_table()
bfs_form_tree(plotter, threshold_test, counter=bfs_counter)
bfsFail_list, bfsPass_list = bfs_counter.fail_counts, bfs_counter.pass_counts
pass_fail_graph(bfsPass_list,bfsFail_list,pass_list, fail_list, pPass_list, pFail_list, threshold_test)
