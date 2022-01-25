from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter
import math

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.
        # go through each iteration
        for i in range(self.iters):
            # dictionary to save the values, different from the one used in calculation
            values = counter.Counter()
            # go through each state, get the best QValue for each state and save it in
            # the dictionary
            # for terimal state set to default 0
            for state in self.mdp.getStates():
                QValue = -math.inf
                for action in self.mdp.getPossibleActions(state):
                    QValue = max(QValue, self.getQValue(state, action))
                if QValue != -math.inf:
                    values[state] = QValue
                else:
                    values[state] = 0
            self.values = values
        # raise NotImplementedError()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    # Following the formula, we calculate Q*(S, A)
    def getQValue(self, state, action):
        QValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            statePrime, probability = transition
            reward = self.mdp.getReward(state, action, statePrime)
            value = self.values[statePrime]
            discount = self.discountRate
            QValue += probability * (reward + discount * value)
        return QValue

    # following the formula, we calculate arg_max_{action in actions} Q(S, A)
    def getPolicy(self, state):
        policy = None
        if state == 'TERMINAL_STATE':
            return policy
        QValue = -math.inf
        for action in self.mdp.getPossibleActions(state):
            QValue = max(QValue, self.getQValue(state, action))
            if QValue == self.getQValue(state, action):
                policy = action
        return policy
