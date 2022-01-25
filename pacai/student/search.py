"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    """
    Implemented based on BFS code below since I couldn't find DFS Psuedocode from my e textbook
    Used the given Stack class to implement a LIFO stack
    """
    startNode = problem.startingState()
    if problem.isGoal(startNode):
        return []

    DFS_Stack = Stack()
    DFS_Stack.push((startNode, []))
    reachedNodes = [startNode]

    while not DFS_Stack.isEmpty():
        currentNode, actions = DFS_Stack.pop()
        for nextNode, action, cost in problem.successorStates(currentNode):
            actionsUpdated = actions + [action]
            if problem.isGoal(nextNode):
                return actionsUpdated
            if nextNode not in reachedNodes:
                DFS_Stack.push((nextNode, actionsUpdated))
                reachedNodes.append(nextNode)

    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # Implemented based on BFS psuedo code from the text book
    # Used the given Queue class to implement FIFO queue
    startNode = problem.startingState()
    if problem.isGoal(startNode):
        return []

    BFS_Queue = Queue()
    BFS_Queue.push((startNode, []))
    reachedNodes = [startNode]

    while not BFS_Queue.isEmpty():
        currentNode, actions = BFS_Queue.pop()
        for nextNode, action, cost in problem.successorStates(currentNode):
            actionsUpdated = actions + [action]
            if problem.isGoal(nextNode):
                return actionsUpdated
            if nextNode not in reachedNodes:
                BFS_Queue.push((nextNode, actionsUpdated))
                reachedNodes.append(nextNode)

    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # Implemented based on BFS code above
    # Used the given priorityQueue class to implement a priorityQueue
    # based on the cost from one node to the next

    startNode = problem.startingState()
    if problem.isGoal(startNode):
        return []

    UCS_pQueue = PriorityQueue()
    UCS_pQueue.push((startNode, [], 0), 0)
    reachedNodes = [startNode]

    while not UCS_pQueue.isEmpty():
        currentNode, actions, pCost = UCS_pQueue.pop()
        for nextNode, action, cost in problem.successorStates(currentNode):
            actionsUpdated = actions + [action]
            costUpdated = pCost + cost
            if problem.isGoal(nextNode):
                return actionsUpdated
            if nextNode not in reachedNodes:
                UCS_pQueue.push((nextNode, actionsUpdated, costUpdated), costUpdated)
                reachedNodes.append(nextNode)

    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # Implemented based on UCS code above
    # Adding heuristic function that adds a value to the cost from one node to next
    # Priorty queue is based on heuristic + cost

    startNode = problem.startingState()
    if problem.isGoal(startNode):
        return []

    UCS_pQueue = PriorityQueue()
    UCS_pQueue.push((startNode, [], 0), 0)
    reachedNodes = [startNode]

    while not UCS_pQueue.isEmpty():
        currentNode, actions, pCost = UCS_pQueue.pop()
        for nextNode, action, cost in problem.successorStates(currentNode):
            actionsUpdated = actions + [action]
            costUpdated = pCost + cost
            heuristicUpdated = costUpdated + heuristic(nextNode, problem)
            if problem.isGoal(nextNode):
                return actionsUpdated
            if nextNode not in reachedNodes:
                UCS_pQueue.push((nextNode, actionsUpdated, costUpdated), heuristicUpdated)
                reachedNodes.append(nextNode)
    raise NotImplementedError()
