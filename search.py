# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util, queue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    
    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm.
    """
    # Initialize a stack for frontier nodes and a set for explored nodes
    frontier = util.Stack()
    explored = set()
    
    # Get the start state
    start_state = problem.getStartState()
    
    # Create a start node with initial state, empty path, and no action
    start_node = (start_state, [])
    
    # Put the start node in the frontier
    frontier.push(start_node)
    
    # Continue searching while frontier is not empty
    while not frontier.isEmpty():
        # Get the next node from the frontier
        current_state, path = frontier.pop()
        
        # Check if we've reached the goal state
        if problem.isGoalState(current_state):
            return path
        
        # Add current state to explored set if not already explored
        if current_state not in explored:
            explored.add(current_state)
            
            # Expand the current state and explore its successors
            # Note: we iterate in reverse to maintain typical DFS behavior 
            # (first pushed will be last explored)
            for next_state, action, _ in problem.expand(current_state)[::-1]:
                # If the next state hasn't been explored
                if next_state not in explored:
                    # Create a new path by adding the current action
                    new_path = path + [action]
                    
                    # Add the new node to the frontier
                    frontier.push((next_state, new_path))
    
    # If no path is found
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    # Initialize a queue for frontier nodes and a set for explored nodes
    frontier = util.Queue()
    explored = set()
    
    # Get the start state
    start_state = problem.getStartState()
    
    # Create a start node with initial state, empty path, and no action
    start_node = (start_state, [])
    
    # Put the start node in the frontier
    frontier.push(start_node)
    
    # Continue searching while frontier is not empty
    while not frontier.isEmpty():
        # Get the next node from the frontier
        current_state, path = frontier.pop()
        
        # Check if we've reached the goal state
        if problem.isGoalState(current_state):
            return path
        
        # Add current state to explored set if not already explored
        if current_state not in explored:
            explored.add(current_state)
            
            # Expand the current state and explore its successors
            for next_state, action, _ in problem.expand(current_state):
                # If the next state hasn't been explored
                if next_state not in explored:
                    # Create a new path by adding the current action
                    new_path = path + [action]
                    
                    # Add the new node to the frontier
                    frontier.push((next_state, new_path))
    
    # If no path is found
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    from util import PriorityQueue

    # Initialize a priority queue for frontier nodes and a dictionary for explored nodes
    frontier = PriorityQueue()
    explored = set()

    # Get the start state
    start_state = problem.getStartState()

    # Push the start state into the frontier with a priority of 0
    frontier.push((start_state, [], 0), 0)  # (state, path, cost)

    # Continue searching while the frontier is not empty
    while not frontier.isEmpty():
        # Pop the node with the lowest cost
        current_state, path, current_cost = frontier.pop()

        # Check if we've reached the goal state
        if problem.isGoalState(current_state):
            return path

        # If the current state has not been explored
        if current_state not in explored:
            # Add it to the explored set
            explored.add(current_state)

            # Expand the current state and explore its successors
            for next_state, action, step_cost in problem.expand(current_state):
                # Compute the total cost for the next state
                total_cost = current_cost + step_cost

                # If the next state has not been explored or is not in the frontier
                if next_state not in explored:
                    # Push the next state with its path and total cost into the frontier
                    frontier.push((next_state, path + [action], total_cost), total_cost)

    # If no path is found
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
