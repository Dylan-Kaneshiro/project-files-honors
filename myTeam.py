# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Actions, Directions
import game
from util import nearestPoint
from util import PriorityQueue

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

def isInOurSide(self, gameState, position):
    """
    Returns True if the given position is in our side of the board.
    """
    width = gameState.data.layout.width
    if self.red:
      return position[0] < width / 2
    else:
      return position[0] >= width / 2
    
def isFood(self, gameState, position):
    """
    Returns True if the given position has food that our agent can eat.
    """
    foodMatrix = self.getFood(gameState)
    x, y = position
    return foodMatrix[x][y]

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """



  def aStarSearch(self, gameState, goal_test):
    """
    A* search algorithm to find the minimum cost path to a goal.
    The cost of being within distance 1 of a ghost is 10, the cost of being within distance 2 of a ghost is 5,
    the cost of being within distance 3 of a ghost is 2, and the cost of all other moves is 1.
    """
    myPos = gameState.getAgentState(self.index).getPosition()
    myPos = (int(myPos[0]), int(myPos[1]))  # Ensure myPos is an integer tuple
    frontier = PriorityQueue()
    frontier.push((myPos, [], 0), 0)  # Add a third element for the cost
    explored = set()
    gridCenter = (gameState.data.layout.width // 2, gameState.data.layout.height // 2)

    # Get the positions of the opposing team's ghosts
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    while not frontier.isEmpty():
      node, actions, totalCost = frontier.pop()
      if goal_test(self, gameState, node):
        return actions
      explored.add(node)
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        dx, dy = Actions.directionToVector(action)
        nextNode = (int(node[0] + dx), int(node[1] + dy))
        if nextNode not in explored and not gameState.hasWall(int(nextNode[0]), int(nextNode[1])):
           # Calculate the distance to the nearest ghost
          distances_to_ghosts = [self.getMazeDistance(nextNode, ghost.getPosition()) for ghost in ghosts]
          distance_to_nearest_ghost = min(distances_to_ghosts) if distances_to_ghosts else float('inf')
          if distance_to_nearest_ghost == 1:
            cost = 30
          elif distance_to_nearest_ghost == 2:
            cost = 20
          elif distance_to_nearest_ghost == 3:
            cost = 10
          else:
            cost = 1
          newCost = totalCost + cost
          heuristic = abs(nextNode[0] - gridCenter[0])  # Horizontal distance to center
          frontier.push((nextNode, actions + [action], newCost), newCost + heuristic)
    return []

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)

    myPos = successor.getAgentState(self.index).getPosition()
    features['distanceToGhost'] = 5

    # Compute distance to the nearest food
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  def chooseAction(self, gameState):
    """
    Picks among actions randomly after computing their values.
    If the agent is carrying food and a ghost is within distance 4,
    it uses the A* search algorithm to calculate a path to the home side.
    """

    if gameState.getAgentState(self.index).numCarrying > 0 and min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), gameState.getAgentState(ghostIndex).getPosition()) for ghostIndex in self.getOpponents(gameState) if gameState.getAgentState(ghostIndex).getPosition() != None], default=float('inf')) <= 4:      # Use A* search to find path to home side
      path = self.aStarSearch(gameState, isInOurSide)
      return path[0]
    elif len(self.getFood(gameState).asList()) > 0:
      path = self.aStarSearch(gameState, isFood)
      return path[0]
    else:
      path = self.aStarSearch(gameState, isInOurSide)
      return path[0]

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def __init__( self, index, timeForComputing = .1 ):
    super().__init__(index, timeForComputing)
    self.lastSighting = None
    
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['offDefense'] = 0
    if myState.isPacman: features['offDefense'] = 1

    # Remove last sighting upon reaching it
    if myPos == self.lastSighting:
      self.lastSighting = None

    # Find positions of invaders that just picked up food
    prevGameState = self.getPreviousObservation()
    thiefPositions = []
    if prevGameState:
      oldFood = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
      curFood = self.getFoodYouAreDefending(gameState).asList()
      for food in oldFood:
        if food not in curFood: # If this food was stolen last turn, add position to known invader positions
          thiefPositions.append(food)
          if self.lastSighting == None or self.getMazeDistance(myPos, food) < self.getMazeDistance(myPos, self.lastSighting): # Keep track of position of last food stolen
            self.lastSighting = food

    # Computes distance to invaders we can see or any invader that just picked up food
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0 or len(thiefPositions) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      dists += [self.getMazeDistance(myPos, pos) for pos in thiefPositions]
      features['invaderDistance'] = min(dists)

    # If no invaders in sight, follow last food disappearance
    elif self.lastSighting != None:
      dist = self.getMazeDistance(myPos, self.lastSighting)
      features['invaderDistance'] = dist

    # Compute "distance to border" - minimum maze distance to any coordinate one away from the border
    border = int(self.getFood(gameState).width // 2)
    if self.red:
      column = border - 1
    else:
      column = border + 1
    borderDists = []
    for y in range(self.getFood(gameState).height):
      possiblePair = (myPos, (column, y))
      if possiblePair in self.distancer._distances: # If this border point is valid in the maze, add maze distance from myPos to this point
        pos1,pos2 = possiblePair
        borderDists.append(self.getMazeDistance(pos1, pos2))
    features['distanceToBorder'] = min(borderDists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'offDefense': -1000, 'invaderDistance': -100, 'distanceToBorder': -10, 'stop': -100, 'reverse': -2}
