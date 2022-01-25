import random
import math

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        score = 0

        # In class discussion, we talked about how checking is a ghost is 2 states away
        # If the ghost is within 2 space after the move, we disencourages it
        # However is the ghost is scared, we encourages it
        for ghost in newGhostStates:
            if distance.manhattan(newPosition, ghost.getPosition()) < 2:
                if ghost.isBraveGhost():
                    score = - 2000
                else:
                    score += 2000

        # If we are on a food pellet, we encourage it
        # We encourage a move based on the combine distance to all the food
        # The less distance the better
        foodList = oldFood.asList()
        if newPosition in foodList:
            score += 500
        else:
            for food in foodList:
                score += 10 / distance.manhattan(food, newPosition)

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # getAction is using Minimax to return the most desirable action
    # based on textbook pseudocode
    def getAction(self, state):
        v = -math.inf
        actions = state.getLegalActions(0)
        actions.remove("Stop")
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v2 = self.maxValue(successorState, self.getTreeDepth())
            if v2 > v:
                a = action
                v = v2
        return a

    # returns the best value chosen by the pac man
    # check each action except for Stop, and choose the best value
    # returned by the advasary(ghosts/min)
    def maxValue(self, state, depth):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        v = -math.inf
        actions = state.getLegalActions(0)
        actions.remove("Stop")
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v2 = self.minValue(successorState, 1, depth)
            if v2 > v:
                v = v2
        return v

    # returns the best value chosen by the ghosts
    # same idea as maxValue, except for that it either checks the value
    # returned by the pacman or the value by other ghosts
    def minValue(self, state, agentID, depth):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        v = math.inf
        actions = state.getLegalActions(agentID)
        for action in actions:
            successorState = state.generateSuccessor(agentID, action)
            if agentID == state.getNumAgents() - 1:
                v2 = self.maxValue(successorState, depth - 1)
            else:
                v2 = self.minValue(successorState, agentID + 1, depth)
            if v2 < v:
                v = v2
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # minimax with alpha and beta variable to save computing time
    # based on textbook pseudocode
    def getAction(self, state):
        v = -math.inf
        actions = state.getLegalActions(0)
        actions.remove("Stop")
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v2 = self.maxValue(successorState, self.getTreeDepth(), -math.inf, math.inf)
            if v2 > v:
                a = action
                v = v2
        return a

    # Same as minimax's maxValue excepts for it returns
    # the value early if the value is greater than or equal to
    # the (newly) passed in beta
    def maxValue(self, state, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        v = -math.inf
        actions = state.getLegalActions(0)
        actions.remove("Stop")
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v2 = self.minValue(successorState, 1, depth, alpha, beta)
            if v2 > v:
                v = v2
                alpha = max(alpha, v2)
            if v >= beta:
                return v
        return v

    # Same as minimax's minValue excepts for it returns
    # the value early if the value is less than or equal to
    # the (newly) passed in alpha
    def minValue(self, state, agentID, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        v = math.inf
        actions = state.getLegalActions(agentID)
        for action in actions:
            successorState = state.generateSuccessor(agentID, action)
            if agentID == state.getNumAgents() - 1:
                v2 = self.maxValue(successorState, depth - 1, alpha, beta)
            else:
                v2 = self.minValue(successorState, agentID + 1, depth, alpha, beta)
            if v2 < v:
                v = v2
                beta = min(beta, v2)
            if v <= alpha:
                return v
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Using expectimax pseudocode from lecture
    def getAction(self, state):
        v = -math.inf
        a = "Stop"
        actions = state.getLegalActions(0)
        actions.remove("Stop")
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v2 = self.value(successorState, self.getTreeDepth(), 1)
            if v2 > v:
                a = action
                v = v2
        return a

    # helper function to recursively calls for pac man state and ghosts states
    def value(self, state, depth, agentID):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        if agentID == 0:
            return self.maxValues(state, depth)
        else:
            return self.expValue(state, depth, agentID)

    # return the pacman's best value
    def maxValues(self, state, depth):
        values = []
        actions = state.getLegalActions(0)
        for action in actions:
            successorState = state.generateSuccessor(0, action)
            v = self.value(successorState, depth - 1, 1)
            values.append(v)
        return max(values)

    # return the ghost's best value based on probability.
    # The probablity is uniformly distributed, and ghosts depends on the
    # expected value of the other ghosts. So I can just add the ghost's value
    # for each action and divide by the number of actions
    def expValue(self, state, depth, agentID):
        actions = state.getLegalActions(agentID)
        v = 0
        probability = 1 / len(actions)
        for action in actions:
            successorState = state.generateSuccessor(agentID, action)
            v += self.value(successorState, depth, (agentID + 1) % state.getNumAgents())
        v = v * probability
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: Instead of adding points, I deduct points. So based on ghost, food, and capsule,
    it can deduct less point.

    Added win/lose state, where their value is infinte since they are the most important.

    Added capsule check. Capsules while not necssary, it is still very valuable since it can allow
    the pacman to eat the ghost for more point or allow pacman easier traversal through the game
    state and collect more point, thus it weights a lot to collect them.

    Improved ghost check. Instead of only caring if ghost is within 2 space, it now loses point
    based on close a ghost within 5 space is getting. Pacman will become more prepared and can
    choose better actions

    Improved food check. Now pacman checks for closest food, and lose more points the further
    away it is. It also checks the entire food list, pacman loses more points the more food that
    is still present. This is weighted the highest since the goal is to win, so we must collect
    the food pellet ASAP
    """
    currentPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    score = 0

    if currentGameState.isLose():
        return -math.inf
    elif currentGameState.isWin():
        return math.inf

    capsuleNum = len(currentGameState.getCapsules())
    score = -50 * capsuleNum

    disGhost = [0]
    for ghost in ghostStates:
        if not ghost.isBraveGhost():
            gDis = distance.manhattan(currentPosition, ghost.getPosition())
            if gDis < 5:
                disGhost.append(gDis)

    score -= 5 * min(disGhost)

    disFood = []
    for food in foodList:
        fDis = distance.manhattan(currentPosition, food)
        disFood.append(fDis)

    score -= 4 * min(disFood)

    score -= 10 * len(foodList)

    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
