from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util import probability
import math
import random
class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Problem is similar to valueIterationAgent. We initialize the dictionary
    for QValue instead of Value, and implement getValue over getQValue. getValue and
    getPolicy implementation also almost identical except I checked for terminal state in
    the beginning and added a check to return a random action for the same value. getAction
    returns random with epsilon chance, and returns policy 1-epsilon chance. Update follows
    the formula in lecture slide to compute Q(S,A) >
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.QValues = counter.Counter()

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.QValues[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)
        if(not actions):
            return 0.0
        bestValue = -math.inf
        for action in actions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        return bestValue

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        policy = None
        if(not actions):
            return policy
        bestValue = -math.inf
        for action in actions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
                policy = action
            elif value == bestValue:
                policy = random.choice([policy, action])

        return policy

    def getAction(self, state):
        if probability.flipCoin(self.getEpsilon()):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        alpha = self.getAlpha()
        discount = self.getDiscountRate()
        QValue = self.getQValue(state, action)
        bestValue = self.getValue(nextState)
        sample = reward + discount * bestValue
        newValue = (1 - alpha) * QValue + alpha * sample
        self.QValues[(state, action)] = newValue

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Initilize the weights dictionary. updated getQValue to
    return w * featureVector. Updated update to set weights based on the
    function given in the instructions>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = counter.Counter()

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass

    def getQValue(self, state, action):
        featureVectors = self.featExtractor().getFeatures(state, action)
        value = 0
        for feature in featureVectors:
            value += self.weights[feature] * featureVectors[feature]
        return value

    def update(self, state, action, nextState, reward):
        alpha = self.getAlpha()
        discount = self.getDiscountRate()
        QValue = self.getQValue(state, action)
        bestValue = self.getValue(nextState)
        featureVectors = self.featExtractor().getFeatures(state, action)
        correction = reward + discount * bestValue - QValue
        for feature in featureVectors:
            self.weights[feature] = (self.weights[feature] + alpha * correction
                                    * featureVectors[feature])
