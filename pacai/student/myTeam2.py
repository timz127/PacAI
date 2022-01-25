from pacai.util import reflection, counter
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions

import random

"""
Useful methods to (maybe) use:
    chooseAction(self, gameState)
    getAction(self, gameState)
    getFood(self, gameState)
    getFoodYouAreDefending(self, gameState)
    getCapsulesYouAreDefending(self, gameState)
    getOpponents(self, gameState)
    getTeam(self, gameState)
    getScore(self, gameState)
    getMazeDistance(self, pos1, pos2)
    getPreviousObservation(self)
    getCurrentObservation
    registerTeam(self, agentsOnTeam)
    observationFunction(self, state)
"""

class AttackerAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        evalue = 'attack'
        myState = gameState.getAgentState(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]

        # invaders= [a for a in enemies if a.getPosition() is not None]
        
        if invaders and not myState.isPacman() and myState.isBraveGhost():
            evalue = 'defend'
        
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        values = [self.evaluate(gameState, a, evalue) for a in actions]
        # print("values: ", values)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evaluate(self, gameState, action, evalue):
        features = self.getFeatures(gameState, action, evalue)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action, evalue):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        myTeam = self.getTeam(successor)

        if self.index != myTeam[0]:
            index = 0
        else:
            index = 1

        otherPos = successor.getAgentState(myTeam[index]).getPosition()

        agentDistance = self.getMazeDistance(myPos, otherPos)
        features['teamDistance'] = 1
        if myState.isPacman() and agentDistance != 0:
            features['teamDistance'] = 1 / agentDistance

        if evalue == 'attack':
            features['successorScore'] = self.getScore(successor)

            foodList = self.getFood(successor).asList()

            # This should always be True, but better safe than sorry.
            myPos = myState.getPosition()
            if (len(foodList) > 0):
                minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
                features['distanceToFood'] = minDistance

            minDist = float("inf")
            for capsule in self.getCapsules(successor):
                minDist = min(minDist, self.getMazeDistance(myPos, capsule))
            # features['capsule'] = minDist
        
            ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            closest = float("inf")
            
            for g in ghosts:
                closest = min(self.getMazeDistance(myPos, g.getPosition()), closest)
                features["closestGhost"] = 1/3
                if closest < 2:
                    if g.isBraveGhost() and not myState.isPacman() and closest != 0:
                        features["closestGhost"] = 1/closest
                        # features['capsule'] = 20
        
            features['capsule'] = len(self.getCapsules(successor))
            

            return features
        
        else:
            # Computes whether we're on defense (1) or offense (0).
            features['onDefense'] = 1
            if (myState.isPacman()):
                features['onDefense'] = 0

            # Computes distance to invaders we can see.
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
            features['numInvaders'] = len(invaders)

            if (len(invaders) > 0):
                closest_dist = float("inf")
                closest_ghost = None
                for ghost in invaders:
                    invader_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if closest_dist > invader_dist : # inf > dist
                        closest_dist = invader_dist
                        closest_ghost = ghost
                features['invaderDistance'] = closest_dist
                features['closestInvader'] = closest_ghost
            else:
                closest_dist = float("inf")
                closest_ghost = None
                for ghost in enemies:
                    enemy_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if  closest_dist > enemy_dist:
                        closest_dist = enemy_dist
                        closest_ghost = ghost
                features['enemyDistance'] = closest_dist
                features['closestEnemy'] = closest_ghost

            if (action == Directions.STOP):
                features['stop'] = 1

            rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
            if (action == rev):
                features['reverse'] = 1

            return features
        

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'closestGhost': -1000,
            'capsule': -40,
            'teamDistance': -7,

            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'enemyDistance': -15,
            'stop': -100,
            'reverse': -2
        }

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.AttackerAgent',
        second = 'pacai.student.myTeam.DefenderAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = AttackerAgent
    secondAgent = AttackerAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
