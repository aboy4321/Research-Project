import itertools
import heapq
from timer import Timer
import math
import time
#import numpy as np
#from matplotlib import pyplot as plt
#import pylab
import pickle
#import sys
#sys.setrecursionlimit(2048)


PLOT_SEARCH_SPACE = True

class ThresholdTest:
    id_counter = 0 # next available id
    def __init__(self, weights, threshold, indices=None, size=None, bounds=None):
        #establishing variables
        self.weights = weights
        self.indices = indices 
        self.threshold = threshold

        # instead of copying weights over and over again,
        # we use the same weights list and just vary the size
        if size is None:
            self.size = len(weights)
        else:
            self.size = size

        # it is faster to update the bounds externally instead of
        # re-computing them each time
        if bounds is None:
            self.bounds = Bounds.from_weights(weights,size)
        else:
            self.bounds = bounds

        # children and parents
        self.lo = None
        self.hi = None
        self.parents = []

        # unique id
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
        weights = self.get_weights()
        if not weights:
            return f"0 &#8805; {self.threshold}"
        #root = " + ".join([f"{weight}&#xb7;<I>X</I><SUB>{var+1}</SUB>" for var, weight in enumerate(weights)])
        root = f"{weights[0]}&#xb7;<I>X</I><SUB>{1}</SUB>"
        for var,weight in enumerate(weights[1:],1):
            if weight > 0:
                root += f" + {weight}&#xb7;<I>X</I><SUB>{var+1}</SUB>"
            else:
                root += f" - {-weight}&#xb7;<I>X</I><SUB>{var+1}</SUB>"
        root = f"{root} &#8805; {self.threshold}"
        return root

    @staticmethod
    def parse(st):
        """Parse a neuron string format"""
        neuron = {}
        for line in st.split('\n'):
            line = line.strip()
            if not line: continue
            field,value = line.split(':')
            field = field.strip()
            value = value.strip()
            neuron[field] = value
        assert "size" in neuron
        assert "threshold" in neuron # or "bias" in neuron
        assert "weights" in neuron

        weights = list(map(int,neuron["weights"].split()))
        indices, weights = ThresholdTest.sort_weights(weights)
        threshold = int(neuron["threshold"])

        return ThresholdTest(weights,threshold, indices=indices)

    @staticmethod
    def read(filename):
        """Read a neuron from file"""
        with open(filename,'r') as f:
            st = f.read()
        return ThresholdTest.parse(st)
    def get_weights(self):
        # try not to use this function
        return self.weights[:self.size]

    def get_last_weight(self):
        # note that weights may be longer than size
        # (this is to avoid re-copying the weights in set_last_input()
        return self.weights[self.size-1]
    @classmethod
    def sort_weights(cls, weights):
        weights = list(enumerate(weights))
        weights = sorted(weights,key=lambda x: abs(x[1]))
        indices,weights = zip(*weights)
        return indices, weights
    # Functions testing for triviality:
    def is_trivial_fail(self):
        lower,upper = self.bounds.get_bounds()
        threshold = self.threshold
        # True if the lower and upper are both less than the threshold
        return lower <= upper < threshold

    def is_trivial_pass(self):
        lower,upper = self.bounds.get_bounds()
        threshold = self.threshold
        # True if both the lower and upper are greater than the threshold
        return threshold <= lower <= upper
  
    # Form_tree (pruned tree) is based off this truth table
    def as_truth_table(self):
        passed = 0
        fail = 0
        pass_list = [0]
        fail_list = [0]

        all_combinations = list(itertools.product([0, 1], repeat=self.size))
        headers = [f"X_{i + 1}" for i in (range(self.size))]
        weights = self.get_weights()

        #iterating through all possible combinations
        for c in all_combinations:
            #sums the weighted inputs from the combinations
            weighted_sum = sum(weight * inputs for weight, inputs in zip(weights, c))
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
        assert self.size > 0
        last_weight = self.get_last_weight()

        # update threshold
        #when values in path are set to 1
        if value == 1:
            new_threshold = self.threshold - last_weight
        #when values in path are set to 0
        elif value == 0:
            new_threshold = self.threshold

        # update bounds
        lb,ub = self.bounds.get_bounds()
        if last_weight > 0:
            ub -= last_weight
        else:
            lb -= last_weight
        bounds = Bounds(lb,ub)

        return ThresholdTest(self.weights, new_threshold, indices=self.indices ,size=self.size-1, bounds=bounds)

    def _all_nodes(self,cache=None,only_internal=False):
        # returns a set containing all nodes induced by a threshold test
        if cache is None: cache = set()
        key = self.id # can be (depth,threshold)
        if key in cache: return cache

        if only_internal:
            # a node is leaf if both lo and hi are None
            # otherwise, we call it an internal node
            if self.lo is None and self.hi is None:
                return cache
        cache.add(key)

        if self.lo is not None:
            cache = self.lo._all_nodes(cache=cache,only_internal=only_internal)
        if self.hi is not None:
            cache = self.hi._all_nodes(cache=cache,only_internal=only_internal)
        return cache

    def node_count(self,only_internal=False):
        return len(self._all_nodes(only_internal=only_internal))

    def tree_node_count(self,only_internal=False):
        if only_internal:
            if self.lo is None and self.hi is None:
                return 0
        count = 1
        if self.lo is not None:
            count += self.lo.tree_node_count(only_internal=only_internal)
        if self.hi is not None:
            count += self.hi.tree_node_count(only_internal=only_internal)
        return count

    def model_count(self,cache=None):
        if cache is None: cache = {}
        if self.id in cache:
            return cache[self.id]

        if self.is_trivial_pass():
            count = 2 ** self.size
        elif self.is_trivial_fail():
            count = 0
        else:
            count = 0
            count += self.lo.model_count(cache=cache)
            count += self.hi.model_count(cache=cache)

        cache[self.id] = count
        return count
    def get_last_idx(self, image):
        pass
    def classify(self, image, pair=(0, 1)):
        current = self
        path = []
        image = image.reshape(28, 28)
        while True:
            if current.is_trivial_pass():
                return pair[0], path
            if current.is_trivial_fail():
                return pair[1], path
            
            var_idx = current.size  
            weight = current.get_last_weight()
            val = 1 if current.get_last_weight() > 0 else 0
            path.append((var_idx, weight,val))
            current = current.set_last_input(val) 

# Class used to measure the upper and lower bounds of the threshold test    
class Bounds():
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    @classmethod
    def from_weights(cls,weights,size):
        lb,ub = 0,0
        for weight in weights[:size]:
            if weight < 0: lb += weight
            if weight > 0: ub += weight
        return cls(lb,ub)

    def upper_bound(self):
        return self.ub

    def lower_bound(self):
        return self.lb

    def get_bounds(self):
        return self.lb,self.ub

    # Measures the differences of the upper and lower bound from the respective threshold
    def gap_size(self, test):

        upper_diff = abs(self.upper_bound() - test.threshold)
        lower_diff = abs(test.threshold - self.lower_bound())

        score_list = [self.lower_bound(), lower_diff, test.threshold,upper_diff, self.upper_bound()]
        return score_list

class NullPlotter:
    def __init__(self): pass
    def add_node(self, node_id, label, color): pass
    def add_edge(self,  parent_id, child_id, label): pass
    def draw_tree(self, test, filename): pass
    def draw_graph(self, test, filename): pass

# Class used for the making of the decision tree, utilizes pygraphviz
class TreePlotter():
    def __init__(self):
        pass

    # Function used to add nodes to the tree:
    def add_node(self, test, node_id):
        label = f"{test}"
        if test.is_trivial_pass():
            color = 'green'
        elif test.is_trivial_fail():
            color = 'red'
        else:
            color = 'black'
        self.graph.add_node(node_id, label=f"<{label}>", shape='box', color=color)

    # Function to add edges to the nodes, from parent to child
    def add_edge(self, test, parent_id, child_id, value):
        style = "solid" if value == 1 else "dashed"
        label = f"<I>X</I><SUB>{test.size + 1}</SUB> = {value}"
        self.graph.add_edge(parent_id, child_id, label=f"<{label}>", style=style)

    # Draws the actual tree and saves the image as a .png file
    def draw_tree(self, test, filename):
        import pygraphviz
        self.graph = pygraphviz.AGraph(strict=True, directed=False)
        self.counter = 0
        self._draw_tree(test)
        self.graph.layout(prog="dot")
        self.graph.draw(filename)

    def _draw_tree(self, test, depth=0, parent_id=None, edge_value=None):
        current_id = f"Node_{depth}_{self.counter}"
        self.counter += 1

        self.add_node(test, current_id)
        if parent_id is not None:
            self.add_edge(test, parent_id, current_id, value=edge_value)

        if test.lo is not None:
            self._draw_tree(test.lo,depth=depth+1,parent_id=current_id,edge_value=0)
        if test.hi is not None:
            self._draw_tree(test.hi,depth=depth+1,parent_id=current_id,edge_value=1)

    # Draws the actual tree and saves the image as a .png file
    def draw_graph(self, test, filename):
        import pygraphviz
        self.graph = pygraphviz.AGraph(strict=True, directed=False)
        cache = {}
        self._draw_graph(test,cache)
        self.graph.layout(prog="dot")
        self.graph.draw(filename)

    def _draw_graph(self, test, cache, depth=0, parent_id=None, edge_value=None):
        current_id = f"Node_{depth}_{test.threshold}"
        key = (depth,test.threshold)
        if key in cache:
            test = cache[key]
        else:
            self.add_node(test, current_id)
            if test.lo is not None:
                self._draw_graph(test.lo,cache,depth=depth+1,parent_id=current_id,edge_value=0)
            if test.hi is not None:
                self._draw_graph(test.hi,cache,depth=depth+1,parent_id=current_id,edge_value=1)
            cache[key] = test

        if parent_id is not None:
            self.add_edge(test, parent_id, current_id, value=edge_value)

# Class used to create an array of pass and fails 
class Counter:
    def __init__(self, test_size):
        self.test_size = test_size
        self.passes = 0
        self.fails = 0
        self.pass_counts = [0]
        self.fail_counts = [0]
        self.count = 0
        self.seen = {}       
        self.start_time = time.time()
        self.count_times = [0]
 
    def is_trivial_and_count_old(self, test):
        total_counts = 2 ** test.size
        self.count_times.append(time.time() - self.start_time)
        if test.is_trivial_pass():
            self.passes += total_counts
            self.pass_counts.append(self.passes)
            self.fail_counts.append(self.fails)
            return True
        if test.is_trivial_fail():
            self.fails += total_counts
            self.fail_counts.append(self.fails)
            self.pass_counts.append(self.passes)

            return True
        return False

    def is_trivial_and_count(self, test):
        total_counts = 2 ** test.size
        self.count_times.append(time.time() - self.start_time)
        if test.is_trivial_pass():
            root_counts = self._propagate_count(test,[total_counts,0])
            self.passes += root_counts[0]
            self.pass_counts.append(self.passes)
            self.fail_counts.append(self.fails)
            return True
        if test.is_trivial_fail():
            root_counts = self._propagate_count(test,[0,total_counts])
            self.fails += root_counts[1]
            self.fail_counts.append(self.fails)
            self.pass_counts.append(self.passes)
            return True
        return False

    def cache_count(self, test, parent_test):
        counts = test._counts
        if counts[0] + counts[1] == 0:
            # nothing to update
            return
        self.count_times.append(time.time() - self.start_time)
        root_counts = self._propagate_count(parent_test,counts)
        self.passes += root_counts[0]
        self.fails += root_counts[1]
        self.pass_counts.append(self.passes)
        self.fail_counts.append(self.fails)
    
    @classmethod
    def _add_counts(cls,a,b):
        a[0] += b[0]
        a[1] += b[1]

    def _propagate_count(self, test, counts):
        from collections import deque
        queue = deque()
        visited_tests = list()
        visited_ids = set()

        # initialize visited list
        visited_tests.append(test)
        visited_ids.add(test.id)

        test._data = counts
        Counter._add_counts(test._counts,counts)

        # assumes ThresholdTest._data is set to zero
        for parent in test.parents:
            Counter._add_counts(parent._data,test._data)
            Counter._add_counts(parent._counts,test._data)
            queue.extend(test.parents)

        while queue:
            test = queue.popleft()

            if test.id in visited_ids: continue
            visited_tests.append(test)
            visited_ids.add(test.id)

            for parent in test.parents:
                Counter._add_counts(parent._data,test._data)
                Counter._add_counts(parent._counts,test._data)
                queue.extend(test.parents)

        count = test._data # this is the count on the root node

        # reset data field
        for test in visited_tests:
            test._data = [0,0]

        return count

    def count_passing_inputs(self, test, key=None):
        if test.is_trivial_pass():
            return 2 ** test.size
        if test.is_trivial_fail():
            return 0
        
        if key and key in self.seen:
            count = self.seen[key]
            return count
        
        count = 0
        weights = test.weights
        threshold = test.threshold
        size = test.size
        all_combinations = itertools.product([0, 1], repeat=size)
        
        for c in all_combinations:
            weighted_sum = sum(w * x for w, x in zip(weights, c))
            if weighted_sum >= threshold:
                count += 1
                
        # Update internal counters
        
        if key:
            self.seen[key] = count
        return count
    
# Function used to create the pruned tree using a depth-first search algorithm
def form_tree(plot, test, parent_id=None, depth=0, counter=None):
    
    # Displaying important values onto the nodes (mainly steps to pass and fail)

    current_id = f"Node_{depth}_{test.id}"
    if test.is_trivial_pass():
        node_color = 'green'
    elif test.is_trivial_fail():
        node_color = 'red'
    else:
        node_color = 'black'

        #plot.add_node(current_id, current_label, color = node_color)

    #if parent_id is not None:
    #plot.add_edge(parent_id, current_id, label=f"x_{test.size + 1}")

    if counter.is_trivial_and_count_old(test):
        return

    # Iterates between the binary inputs of 0 and 1
    for next_value in [1, 0]:
        # Recursion, adds upon the depth of the tree
        reduced_test = test.set_last_input(next_value)
        form_tree(plot, reduced_test,parent_id=current_id, depth=depth+1, counter=counter)

# Improved form_tree function using a best-first search algorithm
# with the use of heaps
# NOTE: Will be going through this function step-by-step

def model_count(test, depth, cache = {}):
    key = (depth, test.threshold, tuple(test.weights))
    if key in cache: return cache[key]

    if test.is_trivial_pass():
        count = 2 ** test.size
    elif test.is_trivial_fail():
        count= 0
    else:
        high_test = test.set_last_input(1)
        high_count = model_count(high_test, depth + 1, cache)

        low_test = test.set_last_input(0)
        low_count = model_count(low_test, depth+1, cache)

        count = high_count + low_count
        cache[key] = count
    return count

def compute_priority_A(test):
    return min(steps_to_pass(test), steps_to_fail(test))

def compute_priority_R(test): # from Richard's paper
    return steps_to_pass(test)

def bfs_form_tree(test, counter=None, priority_f=compute_priority_A):
    # Initializing an empty heap array we will be iterating upon:
    heap = []
    seen = {}
    # Creating and "pushing" our priorities to the heap

    initial_priority = priority_f(test)
    heapq.heappush(heap, (initial_priority, 0, test, None, None))

    # "while" The heap has values within itself
    while heap:
        # Removes the smallest of these variables from the heap and returns it
        priority, depth, test, parent_test, edge_label = heapq.heappop(heap)

        key = (depth, test.threshold)
        if key in seen:
            test = seen[key]

            if parent_test is not None:
                # there must be a parent in this case

                if edge_label == 0: parent_test.lo = test
                else:               parent_test.hi = test
                test.parents.append(parent_test)
                counter.cache_count(test,parent_test)
        else:
            seen[key] = test # Mark this (depth, threshold) combo as seen
            test._data = [0,0]   # initialize temp data
            test._counts = [0,0] #

            if parent_test is not None:
                if edge_label == 0: parent_test.lo = test
                else:               parent_test.hi = test
                test.parents.append(parent_test)

            if counter.is_trivial_and_count(test):
                pass                
            else:
                
                # add children to priority queue
                left = test.set_last_input(0)
                right = test.set_last_input(1)
                left_priority = priority_f(left)
                right_priority = priority_f(right)
                heapq.heappush(heap, (left_priority, depth+1,left, test,0))
                heapq.heappush(heap, (right_priority,depth+1,right,test,1))

                # ^^Function continues until heap is empty^^     
def robust(test, image, label):
    heap = [(0, test, image, [])]
    while heap:
        cost, test, current_image, path = heapq.heappop(heap)
        if label == True:
            if test.is_trivial_fail():
                return path

            if test.is_trivial_pass():
                continue
        else:
            if test.is_trivial_fail():
                continue

            if test.is_trivial_pass():
                return path
        pixel_idx = test.indices[test.size - 1]

        value = int(current_image[pixel_idx])

        for next_value in [0,1]:
            if next_value == value:
                flip_cost = 0
            else:
                flip_cost = 1
            new_image = current_image.copy()

            new_image[pixel_idx] = next_value
            new_test = test.set_last_input(next_value)

            new_path = path + [(pixel_idx, value, next_value)] 
            heapq.heappush(heap, (cost+flip_cost, new_test, new_image, new_path))

def old_bfs(test, counter=None, priority_f=compute_priority_R):
    heap = []
    
    #p = steps_to_pass(test)
    #f = steps_to_fail(test)

    priority = priority_f(test)
    heapq.heappush(heap,(priority, 0, test, None, None))

    while heap:
        priority, depth, test, parent_test, edge_label = heapq.heappop(heap)
        
        if parent_test is not None:
            if edge_label == 0: parent_test.lo = test
            else:               parent_test.hi = test
            test.parents.append(parent_test)
        if counter.is_trivial_and_count_old(test):
            continue
        else:
            left = test.set_last_input(0)
            right = test.set_last_input(1)

            priority = priority_f(left)
            heapq.heappush(heap, (priority, depth + 1, left, test, 0))
            priority = priority_f(right)
            heapq.heappush(heap, (priority, depth + 1, right, test, 1))

# Iterates throught the amount of steps a node is away from becoming trivial
# using the set_last_input to check
def steps_to_pass(test):
    weights = test.weights
    T = test.threshold
    size = test.size

    lb,ub = test.bounds.get_bounds()
    steps = 0
    while True:
        if T <= lb: return steps
        steps += 1
        weight = weights[size-steps]
        if weight < 0:
            # set input to 0
            lb -= weight
        else:
            # set input to 1
            T -= weight
            ub -= weight

def steps_to_fail(test):
    weights = test.weights
    T = test.threshold
    size = test.size

    lb,ub = test.bounds.get_bounds()
    steps = 0
    while True:
        if ub < T: return steps
        steps += 1
        weight = weights[size-steps]
        if weight < 0:
            # set input to 1
            T -= weight
            lb -= weight
        else:
            # set input to 0
            ub -= weight

# Graph showing the upper and lower bounds of the threshold test, and the steps taken to reach that number
# Mainly going to be used to show the efficiency of our algorithm

# The BFS, DFS and raw values are taken as parameters
def pass_fail_graph(bfs_pass, bfs_fail, pass_list, fail_list, pruned_pass, pruned_fail, test):
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
    
    plt.plot(pass_list,color='blue',linestyle='-')
    plt.plot(fail_list,color='red', linestyle='-')
    plt.plot(pruned_fail,color='red' , linestyle= '-')
    plt.plot(pruned_pass,color='blue' , linestyle= '-')
    plt.plot(bfs_pass, color='magenta', linestyle='-')
    plt.plot(bfs_fail,color='purple', linestyle='-') 
    plt.axhline(y=count, linestyle='--', color="black")

    plt.xscale('log')
    plt.show()
def pass_fail_graph2(bfs_pass, bfs_fail, old_bfs_pass, old_bfs_fail,pruned_pass, pruned_fail, test, counter_pruned=None, counter_bfs=None, counter_old=None):
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
    # computing the top value (this is 2^n)
    total = 2 ** test.size
    
    # Flipping the fail_list to start at the top and count down
    pruned_fail = [total - fail for fail in pruned_fail] 
    bfs_fail = [total - fail for fail  in bfs_fail]
    old_bfs_fail = [total - fail for fail in old_bfs_fail]
    
    # Final count of the lists
    count = bfs_fail[-1]
    pruned_counter = counter_pruned.count_times[:len(pruned_pass)]
    plt.plot(pruned_counter,pruned_fail,color='red' , linestyle= ':')
    plt.plot(pruned_counter,pruned_pass,color='blue' , linestyle= ':')
    bfs_counter = counter_bfs.count_times[:len(bfs_pass)]
    plt.plot(bfs_counter,bfs_pass, color='blue', linestyle='-')
    plt.plot(bfs_counter,bfs_fail,color='red', linestyle='-')
    oldbfs_counter = counter_old.count_times[:len(old_bfs_pass)]
    plt.plot(oldbfs_counter,old_bfs_pass, color="blue", linestyle='-.')
    plt.plot(oldbfs_counter,old_bfs_fail, color='red', linestyle='-.')
    plt.axhline(y=count, linestyle='--', color="black")
    plt.xscale("log")
    plt.savefig("plot-log.png")
    plt.xscale("linear")
    plt.savefig("plot-linear.png")
    plt.show()

def bfs_graph(bfs_pass, bfs_fail, test, counter, plot_times=True):
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
    # computing the top value (this is 2^n)
    total = 2 ** test.size
    
    # Flipping the fail_list to start at the top and count down
    bfs_fail = [total - fail for fail  in bfs_fail]
    
    # Makes sure pass_list and flipped fail_list meet at the same endpoint
    
    # Final count of the lists
    count = bfs_pass[-1]
    if plot_times:
        bfs_counter = counter.count_times[:len(bfs_pass)]
    else:
        bfs_counter = list(range(1,len(bfs_pass)+1))
    plt.plot(bfs_counter,bfs_pass, color='blue', linestyle='-')
    plt.plot(bfs_counter,bfs_fail,color='red', linestyle='-')
    plt.axhline(y=count, linestyle='--', color="black")
    plt.xscale("log")
    plt.savefig("plot-log.png")
    plt.xscale("linear")
    plt.savefig("plot-linear.png")
    plt.show()

def visualize_neuron(train_data, test):
    from matplotlib import pyplot as plt

    # initializing array
    image_2d = image.reshape(28, 28)

    path_pixels = []
    path_bits = []
    while not (test.is_trivial_pass() or test.is_trivial_fail()):
        pixel_idx = test.indices[test.size-1]
        path_pixels.append(pixel_idx)
        pixel_value = image[pixel_idx]
        bit = 1 if pixel_value > 0 else 0
        path_bits.append(bit)
        test = test.set_last_input(bit)

    # increasing contrast
    normalized_image = image_2d / image_2d.max() * 0.3 + 0.3

    # adding dark grey background
    overlay = np.full((28,28), 0.2)  
    
    # layering images over one another
    overlay[:] = normalized_image

    # getting set_last_input pixel values
    for pixel_idx, bit in zip(path_pixels, path_bits):
        r, c = divmod(pixel_idx, 28)
        overlay[r, c] = 1.0 if bit == 1 else 0.0  

    # plotting
    plt.imshow(overlay, cmap='gray', vmin=0, vmax=1)
    #plt.show()
    if test.is_trivial_pass():
        return True
    if test.is_trivial_fail():
        return False
def visualize_counterfactual(image, path):
    from matplotlib import pyplot as plt

    normalized = image / image.max() * 0.3 + 0.3
    overlay = normalized.reshape(28, 28).copy()

    for pixel_idx, original_val, flipped_val in path:
        r, c = divmod(pixel_idx, 28)
        if original_val == flipped_val:
            continue
        if flipped_val == 1:
            overlay[r, c] = 1.0  # white
        else:
            overlay[r, c] = 0.0  # black
 
    plt.imshow(overlay, cmap='gray', vmin=0, vmax=1)
    #plt.show()


if __name__ == '__main__':
    EXPERIMENT1 = False
    EXPERIMENT2 = True

    if EXPERIMENT1:
        pickle_filename = "experiment1.pickle"

        n = 8
        weights = [ 2**x for x in range(n) ] + [ -2**x for x in range(n) ]
        weights = sorted(weights,key=lambda x: abs(x))
        threshold = 0
        test = ThresholdTest(weights, threshold)


        with Timer("old_bfs (new heuristic)"):
            counter_A = Counter(test.size)
            old_bfs(test,counter=counter_A,priority_f=compute_priority_A)

        with Timer("old_bfs (old heuristic)"):
            counter_R = Counter(test.size)
            old_bfs(test,counter=counter_R,priority_f=compute_priority_R)

        data = { 'counter_A': counter_A,
                 'counter_R': counter_R }
        with open(pickle_filename,'wb') as f:
            pickle.dump(data,f)

        #plot_start()
        #plot_one(counter_A,plot_times=False,linestyle='-')
        #plot_one(counter_R,plot_times=False,linestyle=':')
        #plot_end()
        exit()

    if EXPERIMENT2:
        i, j = 0, 1 

        filename = f"data/digits/neuron-{i}-{j}.neuron"
        threshold_test = ThresholdTest.read(filename)
        pair = (i, j)
        #true_digit = threshold_test.classify(pair = pair)
        # Print the decision path
        if PLOT_SEARCH_SPACE: plotter = TreePlotter()
        else:                 plotter = NullPlotter()
        with Timer("dfs"):
            counter = Counter(threshold_test.size)
            form_tree(plotter, threshold_test, counter=counter)
            pFail_list, pPass_list = counter.fail_counts, counter.pass_counts
        """
        with Timer("truth table"):
            pass_list, fail_list = threshold_test.as_truth_table()
            pass_list, fail_list = [pPass_list[-1]],[pFail_list[-1]]
            pass_list, fail_list = threshold_test.as_truth_table()
            pass_list, fail_list = [pPass_list[-1]],[pFail_list[-1]]
        """
        with Timer("old_bfs"):
            old_bfs_counter = Counter(threshold_test.size)
            old_bfs(threshold_test, counter=old_bfs_counter)
            old_bfsFail_list, old_bfsPass_list = old_bfs_counter.fail_counts, old_bfs_counter.pass_counts
        with Timer("bfs"):
            bfs_counter = Counter(threshold_test.size)
            bfs_form_tree(threshold_test, counter=bfs_counter)
            bfsFail_list, bfsPass_list = bfs_counter.fail_counts, bfs_counter.pass_counts  

        #plotter.draw_tree(threshold_test, filename="threshold_tree.png")
        #plotter.draw_graph(threshold_test, filename="threshold_graph.png")
        #print(f"Graph node formula:                {threshold * (n - threshold + 1)}")
        #print(f"Tree node formula:                   {math.comb(n+1, threshold) - 1}")
        #print(f"tree node count (all):      {threshold_test.tree_node_count(only_internal=False)}")
        #print(f"tree node count (internal): {threshold_test.tree_node_count(only_internal=True)}")
        #print(f"model count: {threshold_test.model_count()}")

        #pass_fail_graph(bfsPass_list,bfsFail_list,pass_list, fail_list, pPass_list, pFail_list, threshold_test)
        pass_fail_graph2(bfsPass_list, bfsFail_list, old_bfsPass_list, old_bfsFail_list, pPass_list, pFail_list, threshold_test, counter_pruned=counter,counter_bfs=bfs_counter, counter_old=old_bfs_counter)
        bfs_graph(bfsPass_list, bfsFail_list, threshold_test, bfs_counter)

        import numpy as np
        from matplotlib import pyplot as plt

        image_file= f"data/csv/train-{i}-{j}.txt"
        image_data = np.loadtxt(image_file, delimiter=",")

        #file:///home/aidanboyce/work/projects/Research-Project/index.html
        output_dir = "output"
        f = open(f"index.html",'w')
        for image_index in range(20):
            image = image_data[image_index, :-1]
            label = visualize_neuron(image, threshold_test)
            print(f"{label} neuron: {image_index}")
            neuron_filename = f"{output_dir}/neuron-{image_index}.png"
            plt.savefig(neuron_filename)
            path = robust(threshold_test,image, label)
            visualize_counterfactual(image, path)
            counter_filename = f"{output_dir}/counter-{image_index}.png"
            plt.savefig(counter_filename)
            f.write(f'label: {pair[int(label)]}')
            f.write(f'<img src="{neuron_filename}">')
            f.write(f'<img src="{counter_filename}"><br><br>\n')
        f.close()
