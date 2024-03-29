# importing "copy" for copy operations 
import copy
import sys
import math
import time
import traceback
#import resource

# declare all the yards to be used
yard_1 = [[1,2], [1,3], [3,5], [4,5], [2,6], [5,6]]
yard_2 = [[1,2], [1,5], [2,3], [2,4]]
yard_3 = [[1,2], [1,3]]
yard_4 = [[1,2], [1,3], [1,4]]
yard_5 = [[1,2], [1,3], [1,4]]
# Declare all the initial states of each of the yards to be used
s_init_state = ['*','a','b']

init_state_1 = ['*', 'e', '', 'bca', '', 'd']
init_state_2 = ['*', 'd', 'b', 'ae', 'c']
init_state_3 = ['*', 'a', 'b']
init_state_4 = ['*', 'a', 'bc', 'd']
init_state_5 = ['*', 'a', 'cb', 'd']
# Declare all the end(goal) states of each of the yards
s_end_state = ['*ab','','']

end_state_1 = ['*abcde', '', '', '', '', '']
end_state_2 = ['*abcde', '', '', '', '']
end_state_3 = ['*ab', '', '']
end_state_4 = ['*abcd', '', '', '']
end_state_5 = ['*abcd','','','']

# Define a class for a Track
class Track:
    def __init__(self, track_num, cars):
        self.track_num = track_num
        self.cars = cars
        self.index = track_num - 1
    # END  __init__(self, track_num, cars)
# END Track

# Turns a state into a list of Track classes
def getTrackList(state):
    track_list = []
    track_num = 1
    for track in state:
        t = Track(track_num, track)
        track_list.append(t)
        track_num+=1 # incrememnt the counter
    return track_list
# END getTrackList(state)

def printState(state): 
    counter = 1
    for x in state:
        print(str(counter) + ': ' + str(x))
        counter += 1
# END printState(state)

# Return true if either track contains the engine, otherwise return false
def checkForEngine(track_pair, state):
    # track_pair[0] holds all cars in the left track and track_pair[1] holds all cars in the right track
    left = track_pair[0]-1      # -1 becasuse index of list starts at 0
    right = track_pair[1]-1     # -1 becasuse index of list starts at 0
    track_cars_pair = [state[left], state[right]]
    for track in track_cars_pair:
        for car in track or []:
            if car == '*':
                return True
    return False
# END checkForEngine(track_pair, state)

def findMoveLeft(yard, state, track_num):
    left_actions = [] # A list of actions where the given state can move left
    # Check if there is a pair of tracks in 'yard' where 'track' is on the right

    # Look through the yard list
    for pair in yard:
        # Check if there any cars in the track
        if state[pair[1]-1] != '':  # -1 is used because the index starts at 0
            # Check if the track number is found on the left in the yard 
            if pair[1] == track_num:
                # Check if either track in the pair has the engine
                if checkForEngine(pair, state) == True:
                    # Create and append an action
                    action = ['LEFT', pair[1], pair[0]] # (DIRECTION FROM-TRACK TO-TRACK) * Transposition of x and y *
                    left_actions.append(action)
    return left_actions
# END findMoveLeft(yard, state, track_num)

def findMoveRight(yard, state, track_num):
    right_actions = [] # A list of actions where the given state can move right
    # Check if there is a pair of tracks in 'yard' where 'track' is on the right

    # Look through the yard list
    for pair in yard:
        # Check if there any cars in the track
        if state[pair[0]-1] != '': # -1 is used because the index starts at 0
            # Check if the track number is found on the right in the yard 
            if pair[0] == track_num:
                # Check if either track in the pair has the engine
                if checkForEngine(pair, state) == True:
                    # Create and append an action
                    action = ['RIGHT', pair[0], pair[1]] # (DIRECTION FROM-TRACK TO-TRACK)
                    right_actions.append(action)
    return right_actions
# END findMoveRight(yard, state, track_num)

# Consumes a yard connectivity list and a state
# Produces a list of all possible actions
def possible_actions(yard, state):
    actions = [] # List of all possible actions for this yard in this state
    # Loop through all tracks in the yard
    tracks = getTrackList(init_state_1)
    for track in tracks:
        new_left_actions = findMoveLeft(yard, state, track.track_num)
        new_right_actions = findMoveRight(yard, state, track.track_num)
        # If new actions were found, append them to the 
        if len(new_left_actions) > 0:
            for a in new_left_actions:
                actions.append(a)
        if len(new_right_actions) > 0:
            for a in new_right_actions:
                actions.append(a)
    return actions
# END possible_actions(yard, state)

# Pushes a car to the front of a track
def push(track, car):
    if track == None:
        track = [str(car)]
    else:
        track[0] = car + track[0] # insert the car at the head of the track
    return track
# END push(track, car)

# Consumes an action and a state
# returns a new state(the state after the given action has occured)
def result(action, state):
    test = 0 # turn this on to allow printing for test
    # Make a copy of the state
    new_state = copy.deepcopy(state)
    # Parse the action 
    direction = action[0]
    from_track = action[1]
    to_track = action[2]

    # Error checking, from_track should never be empty
    if state[from_track-1] == None:
        print("\nERROR: from_track is empty\n")
        sys.exit()
    # Error checking, from_track should never be empty
    if state[from_track-1] == '':
        print(traceback.print_stack())
        print("\nERROR: from_track is empty\n")
        sys.exit()

    if test == 1:
        print(new_state[from_track-1])
        print(new_state[to_track-1])

    # We can assume the action is valid to take bc we are the only actor
    if direction == 'LEFT':
        # Get the car from the from_track
        from_car = new_state[from_track-1][0]
        new_state[from_track-1] = new_state[from_track-1][1:]
        # Check if the to_track is empty
        if new_state[to_track-1] == None:
            new_state[to_track-1] = str(from_car)
        # If its not, append the from_car to the to_track
        else:
            new_state[to_track-1] += from_car # Pop off the first(leftmost) element and append that to the end of the to_track

    elif direction == 'RIGHT':
        # Get the car from the from_track
        from_car = new_state[from_track-1][-1:]
        new_state[from_track-1] = new_state[from_track-1][:-1]
        # Check if the to_track is empty
        if new_state[to_track-1] == None:
            new_state[to_track-1] = str(from_car) # If so, set the from_car as its only element
        # If its not, use my push function to push it to the beginning of the to_track
        else:
            new_state[to_track-1] =  from_car + new_state[to_track-1]

    # Convert empty lists('[]') to None to match tests
    if new_state[from_track-1] == []:
            new_state[from_track-1] = None

    if test == True:
        print(new_state[from_track-1])
        print(new_state[to_track-1])

    return new_state
# END result(action, state)

# Consumes a State and Yard, produces a list of all states that can be reached
def expand(yard, state):
    expanded_states = []
    # Get a list of all actions
    all_actions = possible_actions(yard, state)
    # Add each new state that can be reached
    for action in all_actions:
        expanded_states.append(result(action, state))
    return expanded_states
# END expand(state, yard)

# Node class which will be used to make the graph
class Node:

    def __init__(self, yard, state, parent):
        self.yard = yard
        self.state = state
        self.parent = parent
        self.child_node_list = [] # A list of states available from the given state
        self.action = None # the action taken to get into this state
        self.g = sys.maxsize # Cost form start
        self.h = sys.maxsize # Prediction of cost to end
        self.f = self.g + self.h # Total cost is a combination of g and h
    # END __init__(self, yard, state, parent)
        
    def fill_child_node_list(self):
        self.child_node_list = []
        possible_states = expand(self.yard, self.state) # List of all possible reachable states
        for s in possible_states:
            new_node = Node(self.yard, s, self)
            self.child_node_list.append(new_node)
    # END __init__(self, state, parent)

    def calc_f(self, end_state):
        self.h = heruristic2(self.state, end_state) # get_score is the alternative heuristic
        self.f = self.g + self.h
        return (self.h + self.g)
    # END calc_F(self, end_state)

    def getF(self):
        return (self.g + self.h)
    # END getF(self)
# END Node

def check_list(state, state_list):
    for s in state_list:
        if s == state:
            return True
    return False
# END check_list(state, state_list)

# Global variable used to keep track of all expanded_states in a recursive call
expanded_states = []

# Iterative Deepening Search
def IDS(src, target, max_depth):
    global expanded_states
    # src is a Node
    depth = 1
    action_path = []
    found = False
    while found == False and depth <= max_depth :
        # reset the expanded_states list before running another iteration of DLS
        expanded_states = []
        final_node = DLS(src, target, depth)
        if final_node != None:
            return final_node
        depth += 1 # Increment the depth
    return None
# END IDS(src, target, max_depth)

# Remove action_path and find the path by going up through the nodes parents
# Return the Node or None if not found
# Depth Limited Search
def DLS(src, target, limit):
    global expanded_states
    # Check if the target has been found
    if src.state == target:
        # Use src.action to build an ordered list of action to reach the goal
        action_path = [str(src.action)]
        return src
    
    # Check if the max depth has been reached
    if limit <= 0:
        return None
    
    # If not fill out all children nodes of this node
    src.fill_child_node_list()
    for node in src.child_node_list:
        # Check if the node has already been expanded(not in the expanded list)
        if not check_list(node.state, expanded_states):
            # Add this state to the expanded_nodes list to prevent checking it again
            expanded_states.append(node.state)
            # Store the answer in a variable
            answer_node = DLS(node, target, limit-1)
            if answer_node != None:
                return answer_node
    return None
# END DLS(src, target, limit) 

# Returns what action got you form the start_start to the end_state
def findAction(yard, start_state, end_state):
    # Get all actions you can take from the given state
    actions = possible_actions(yard, start_state)
    for action in actions:
        if result(action, start_state) == end_state:
            return action
    return None
# findAction(yard, start_state, end_state)

# Will be using an iterative-deepening search for my blind search
def blind_search(yard, init_state, goal_state):
    root = Node(yard, init_state, None)
    root.fill_child_node_list()
    final_node = IDS(root, goal_state, 100)
    return expand_answer(final_node, yard)
# END blind_search(yard, init_state, goal_state)

# Got this idea from Jimmy Scripchuk 
# This is the heuristic I used
def heruristic2(state, end_state):
    state_car_string = ""
    end_state_car_string = ""
    for track in state:
        if track != None:
            for car in track:
                if car != None:
                    state_car_string += car
    for track in end_state:
        if track != None:
            for car in track:
                if car != None:
                    end_state_car_string += car
    total_dist = 0
    # Check the total distance that each car is out of place and add them up
    for car_index in range(len(end_state_car_string)):
        for i in range(len(state_car_string)):
            if end_state_car_string[car_index] == state_car_string[i]:
                total_dist += abs(i - car_index)
    return total_dist
# END heruristic2(state, end_state)

# Takes in a list of nodes and returns the one with the smallest f
def get_smallest_f(node_list, end_state):
    # Initiate the smallest node as the first node to start
    smallest_node = node_list[0]
    # Check through every node in the node_list
    for node_num in range(len(node_list)):
        # Calculate the f value, using our heuristic for each node in the nodelist
        node_list[node_num].calc_f(end_state)
        # Check if it is less than the current f value
        if node_list[node_num].f < smallest_node.f:
            # If so make it the smallest node
            smallest_node = node_list[node_num]
    # Return the smallest node
    return smallest_node
# DEF get_smallest_f

# Return the open_list in its properly edited form...
def check_open_list(open_list, node):
    # Look through each node in open list
    for open_node in open_list:
        if open_node.state == node.state:
            # If this nodes f is less than the other nodes f, update open_node to match this node
            if node.f < open_node.f:
                open_node.g = node.g
                open_node.h = node.h
                open_node.f = node.f
                open_node.parent = node.parent

    for open_node in open_list:
        # If a node with a matching state was found, check its f...
        if open_node.state == node.state:
            # If the state already exists and that node has a smaller f, return False, meaning dont add it
            if open_node.f < node.f:
                return False
            # If the node matches the state but is less than the state, you need to update that nodes state
            else:
                return True
    # Return the open_list with all the possible changes
    return open_list
# END check_open_list(open_list, node)

# Check if the node needs to be added to the open_list, only if it is not in the open_list, but update it if it is necessary
def add_node_check(open_list, node):
    for open_node in open_list:
        # If the state matches
        if node.state == open_node.state:
            # Check the f values
            if node.getF() < open_node.getF():
                # node in open_list needs to be updated
                open_node.g = node.g
                open_node.h = node.h
                open_node.f = node.f
                open_node.parent = node.parent
                # Since the node in the open_list has been updated, no reason to add the bn
                return False
            else:
                return False
    # If there is no open_node with a matching state in open_list
    return True
# END add_node_check(open_list, node)

def update_open_list(open_list, node):
    removed_node = None
    for open_node in open_list:
        # If the state matches
        if node.state == open_list.state:
            # Check the f values
            if node.f < open_node.f:
                # node in open_list needs to be updated
                removed_node = open_node
# END update_open_list(open_list, node)

def check_closed_list(closed_list, node):
    for closed_node in closed_list:
        if closed_node.state == node.state:
            if node.getF() >= closed_node.getF():
                return False
    return True
# END check_closed_list(closed_list, node)

#if a node with the same position as successor is in the OPEN list which has a lower f than successor, skip this successor
# If a new_node with the same state is in the open list, check if the f value of the node in the open list is lower than the new_node.f
    # If it is, update the node in the open_list to match the new_node
    # IF the node in the open_list has a lower f value, disreguard this new_node
# If a new_node with the same state is in the closed list, disreguard this new_node
def a_star_search(yard, init_state, goal_state):
    open_list = []
    closed_list = []
    root = Node(yard, init_state, None)
    root.g = 0 
    root.fill_child_node_list()
    open_list = [root] # Can't append to an empty list, so do this instead
    while len(open_list) != 0:
        # Find node in open_list with smallest f
        node = get_smallest_f(open_list, goal_state)
        open_list.remove(node)      # Remove that element from the open_list
        closed_list.append(node)    # Append that element to the closed_list
        # Check if this node is the goal_state
        if node.state == goal_state:
            return node

        # Fill the chosen node's children
        node.fill_child_node_list()
        # For each child of the chosen node
        for child_node in node.child_node_list:
            # Update its g value to be one more than its parent
            child_node.g = node.g + 1
            # Check if the node exists already in closed_list
            # Check if the node even needs to be added to the list
            if add_node_check(open_list, child_node) and check_closed_list(open_list, child_node):
                # If passed all checks, append this node to the open_list and iterate
                open_list.append(child_node)  
        # Append the node to the closed_list 
        closed_list.append(node)     
# END a_star_search(yard, init_state, goal_state)

# Expands the answer node, and returns the list of actions taken to get there
def expand_answer(node, yard):
    action_list = []
    state_list = []
    while node.parent != None:
        # Get the action taken to get from parent to current node
        action = findAction(yard, node.parent.state, node.state)
        # Push the action onto the list
        action_list.insert(0,action)
        state_list.append(node.state)
        # Move back up the route through the parent
        node = node.parent
    return action_list
# END expand_answer(node)

# ***************************************************************************************#

def possible_action_print(yard, state, yard_num):
    print("\nPossible actions for yard " + str(yard_num) + ", state: " + str(state))
    print(possible_actions(yard, state))
# END possible_action_print(yard, state, yard_num)

print("PROBLEM 1")
possible_action_print(yard_1, init_state_1, 1)
possible_action_print(yard_1, end_state_1, 1)
possible_action_print(yard_2, init_state_2, 2)
possible_action_print(yard_2, end_state_2, 2)
possible_action_print(yard_5, init_state_5, 5)
possible_action_print(yard_5, end_state_5, 5)

# TEST PROBLEM 2

a1 = ['LEFT', 2, 1]
a1_sol = ['*a', '', 'b']
assert result(a1, s_init_state) == a1_sol

a2 = ['RIGHT', 1, 2]
a2_sol = ['', '*a', 'b']
assert result(a2, s_init_state) == a2_sol

state = ['*','']
a2_sol = ['', '*']
assert result(a2, state) == a2_sol

state = ['','*']
a2_sol = ['*', '']
assert result(a1, state) == a2_sol


print("Problem 2 is correct")

# TEST PROBLEM 3

expanded_states = expand(yard_3, s_init_state)
expanded_states_sol = [['','*a','b'],
                       ['','a','*b'],
                       ['*a','','b'],
                       ['*b','a','']]
assert expanded_states == expanded_states_sol

print("Problem 3 is correct")

# TEST PROBLEM 4

# Yard 3

blind_start = time.process_time()
print("Running blind search on yard 3")
blind_search(yard_3, init_state_3, end_state_3)
blind_end = time.process_time()
print("Blind search on yard 3 took " + str(blind_end - blind_start) + " seconds")

# Yard 4

blind_start = time.process_time()
print("Running blind search on yard 4")
blind_search(yard_4, init_state_4, end_state_4)
blind_end = time.process_time()
print("Blind search on yard 4 took " + str(blind_end - blind_start) + " seconds")

# Yard 5

blind_start = time.process_time()
print("Running blind search on yard 5")
blind_search(yard_5, init_state_5, end_state_5)
blind_end = time.process_time()
print("Blind search on yard 5 took " + str(blind_end - blind_start) + " seconds")

# Yard 2

blind_start = time.process_time()
print("Running blind search on yard 2")
blind_search(yard_2, init_state_2, end_state_2)
blind_end = time.process_time()
print("Blind search on yard 2 took " + str(blind_end - blind_start) + " seconds")


def search_space_size(c, t):
    # use nPr formula for columns
    num_columns = math.factorial(c) # this is the reduction of c!/(c-c)!
    # use nCr formula for rows
    n = c + t -1    # possible places to choose froms
    r = c           # number chosen
    num_rows = math.factorial(n)/(math.factorial(r) * math.factorial(n-r))
    return num_rows * num_columns
# END search_space_size(c, t)

# PROBLEM 5 ANSWER
# nPr = P(n,r) = n!/(n-r)!
# nCr = C(n,r) = n!/(r!(n-r)!)
#   I calculated this by drawing out the possibilities in a sort of grid pattern
#   Each column represents each possible initial state  
#       Initial states start with all cars on the first track (*ab, null, null) for example
#       All reachable states from this state are the rows of this column 
#   number of columns = P(c,c) = n!/(n-r)!
#       this is a special case where n = r so nPr(c,c) = c! 
#   number of rows = C(c+t-1, c) = n!/
#   total = columns * rows = c! * (c+t-1)!/(c!(c+t-1-c)!)
 
print("\nPROBLEM 5 ANSWERS")
print("Search space (2 cars 2 tracks): " + str(search_space_size(2, 2)))
print("Search space (2 cars 3 tracks): " + str(search_space_size(2, 3)))
print("Search space (3 cars 2 tracks): " + str(search_space_size(3, 2)))
print("Search space (5 cars 5 tracks): " + str(search_space_size(5, 5)))

# TEST PROBLEM 6
print("\nPROBLEM 6")

# Heuristic
#   H = The total distance that each car is out of place and add them up 
#   This is concatonating the state list and comparing it to the concatonation of the goal_state list

# This allows easy testing of different tracks
def run_a_star(yard_num, init_state, end_state):
    yard = None
    # select the required yard
    if yard_num == 1:
        yard = yard_1
    elif yard_num == 2:
        yard = yard_2
    elif yard_num == 3:
        yard = yard_3
    elif yard_num == 4:
        yard = yard_4
    elif yard_num == 5:
        yard = yard_5
    
    # Start the timer
    a_start = time.process_time()

    # Run the search
    solution_node = a_star_search(yard, init_state, end_state)

    # Expand the solution
    solution_sequence = expand_answer(solution_node, yard)

    # End the timer
    a_end = time.process_time()

    # Print the solution as well as the elapsed time
    print("\nSolution sequence for yard " + str(yard_num))
    print(solution_sequence)
    print("A* took " + str(a_end-a_start) + " seconds for yard " + str(yard_num))
# END run_a_star(yard_num, init_state, end_state)

# Run/print a_star for tracks 3-5
run_a_star(3, init_state_3, end_state_3)
run_a_star(4, init_state_4, end_state_4)
run_a_star(5, init_state_5, end_state_5)

# Run/print a_star for track 2
run_a_star(2, init_state_2, end_state_2)