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

class TopAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        evalue = 'attack'
        myState = gameState.getAgentState(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        
        if invaders and not myState.isPacman() and myState.isBraveGhost():
            evalue = 'defend'
        
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        values = [self.evaluate(gameState, a, evalue) for a in actions]
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

        if evalue == 'attack':
            features['successorScore'] = self.getScore(successor)

            foodList = self.getFood(successor).asList()
            topFoods = []
            if (len(foodList) > 0):
                topFoodY = max([foodPos[1] for foodPos in foodList])
                for food in foodList:
                    if food[1] == topFoodY: topFoods.append(food)
                features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in topFoods])
        

            ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            closest = float("inf")
            
            for g in ghosts:
                # closest = min(self.getMazeDistance(myPos, g.getPosition()), closest)
                dist_to_g = self.getMazeDistance(myPos, g.getPosition())
                if dist_to_g < closest:
                    closest = dist_to_g
                    closest_g = g

            if closest_g.isBraveGhost() and myState.isPacman() and closest != 0:
                features["closestGhost"] = 1/closest
            elif closest_g.isBraveGhost() and myState.isPacman() and closest == 0:
                features["closestGhost"] = 10
        
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
                for ghost in invaders:
                    invader_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if invader_dist == 0:
                        closest_dist = -1000
                        break
                    if closest_dist > invader_dist : closest_dist = invader_dist
                features['invaderDistance'] = closest_dist
            else:
                closest_dist = float("inf")
                for ghost in enemies:
                    enemy_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if  closest_dist > enemy_dist: closest_dist = enemy_dist
                features['enemyDistance'] = closest_dist

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
            'closestGhost': -7,
            'capsule': -40,

            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'enemyDistance': -15,
            'stop': -100,
            'reverse': -2
        }

class BottomAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        evalue = 'attack'
        myState = gameState.getAgentState(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        
        if invaders and not myState.isPacman():
            evalue = 'defend'
        
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        values = [self.evaluate(gameState, a, evalue) for a in actions]
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

        if evalue == 'attack':
            features['successorScore'] = self.getScore(successor)

            foodList = self.getFood(successor).asList()
            botFoods = []
            if (len(foodList) > 0):
                botFoodY = min([foodPos[1] for foodPos in foodList])
                for food in foodList:
                    if food[1] == botFoodY: botFoods.append(food)
                features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in botFoods])

            '''
            capDist = float("inf")
            for capsule in self.getCapsules(successor):
                capDist = min(capDist, self.getMazeDistance(myPos, capsule))
            if capDist == 0:
                features['capsule'] = 10
            '''

            ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            closest = float("inf")
            closest_g = None
            for g in ghosts:
                # closest = min(self.getMazeDistance(myPos, g.getPosition()), closest)
                dist_to_g = self.getMazeDistance(myPos, g.getPosition())
                if dist_to_g < closest:
                    closest = dist_to_g
                    closest_g = g
                '''
                features["closestGhost"] = 1/3
                if closest < 2:
                    if g.isBraveGhost() and not myState.isPacman() and closest != 0:
                        features["closestGhost"] = 1/closest
                    elif not g.isBraveGhost() and myState.isPacman() and closest != 0:
                        features["closestGhost"] = 1/closest
                    elif not g.isBraveGhost() and myState.isPacman() and closest == 0:
                        features["closestGhost"] = 1/(closest + 1)
                else:
                    features["closestGhost"] = 0
                '''
            if closest_g.isBraveGhost() and myState.isPacman() and closest != 0:
                features["closestGhost"] = 1/closest
            elif closest_g.isBraveGhost() and myState.isPacman() and closest == 0:
                features["closestGhost"] = 10

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
                for ghost in invaders:
                    invader_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if invader_dist == 0:
                        closest_dist = -1000
                        break
                    if closest_dist > invader_dist : closest_dist = invader_dist
                features['invaderDistance'] = closest_dist
            else:
                closest_dist = float("inf")
                for ghost in enemies:
                    enemy_dist = self.getMazeDistance(myPos, ghost.getPosition())
                    if  closest_dist > enemy_dist: closest_dist = enemy_dist
                features['enemyDistance'] = closest_dist

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
            'closestGhost': -3,
            'capsule': -40,

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

    firstAgent = TopAgent
    secondAgent = BottomAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
