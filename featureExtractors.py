# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math
from util import getDistance, getAngle

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

def threeVSTwoKeepawayFeatures(state, size):
  C = (0.5 * size, 0.5 * size)
  ball = state[0][:2]
  
  keepersId = util.sortByDistances(state[1:4], ball)
  takersId = util.sortByDistances(state[4:], ball)
  keepersId = map(lambda _: _+1, keepersId)
  takersId = map(lambda _: _+4, takersId)

  K1 = state[keepersId[0]]
  K2 = state[keepersId[1]]
  K3 = state[keepersId[2]]
  T1 = state[takersId[0]]
  T2 = state[takersId[1]]

  # get features as a list of real numbers
  feats = [getDistance(K1, C),\
           getDistance(K1, K2),\
           getDistance(K1, K3),\

           getDistance(K1, T1),\
           getDistance(K1, T2),\

           getDistance(K2, C),\
           getDistance(K3, C),\

           getDistance(T1, C),\
           getDistance(T2, C),\

           min(getDistance(K2, T1), getDistance(K2, T2)),\
           min(getDistance(K3, T1), getDistance(K3, T2)),\
           
           min(getAngle(K2, K1, T1), getAngle(K2, K1, T2)),\
           min(getAngle(K3, K1, T1), getAngle(K3, K1, T2))
          ]
  return feats

def fourVSThreeKeepawayFeatures(state, size):
  C = (0.5 * size, 0.5 * size)
  ball = state[0][:2]
  
  keepersId = util.sortByDistances(state[1:4], ball)
  takersId = util.sortByDistances(state[4:], ball)
  keepersId = map(lambda _: _+1, keepersId)
  takersId = map(lambda _: _+4, takersId)

  K1 = state[keepersId[0]]
  K2 = state[keepersId[1]]
  K3 = state[keepersId[2]]
  K4 = state[keepersId[3]]
  T1 = state[takersId[0]]
  T2 = state[takersId[1]]
  T3 = state[takersId[2]]

  # get features as a list of real numbers
  feats = [getDistance(K1, C),\
           getDistance(K1, K2),\
           getDistance(K1, K3),\
           getDistance(K1, K4),\

           getDistance(K1, T1),\
           getDistance(K1, T2),\
           getDistance(K1, T3),\

           getDistance(K2, C),\
           getDistance(K3, C),\
           getDistance(K4, C),\

           getDistance(T1, C),\
           getDistance(T2, C),\
           getDistance(T3, C),\

           min([getDistance(K2, T1), getDistance(K2, T2), getDistance(K2, T3)]),\
           min([getDistance(K3, T1), getDistance(K3, T2), getDistance(K3, T3)]),\
           min([getDistance(K4, T1), getDistance(K4, T2), getDistance(K4, T3)]),\
           
           min([getAngle(K2, K1, T1), getAngle(K2, K1, T2), getAngle(K2, K1, T3)]),\
           min([getAngle(K3, K1, T1), getAngle(K3, K1, T2), getAngle(K3, K1, T3)]),\
           min([getAngle(K4, K1, T1), getAngle(K4, K1, T2), getAngle(K4, K1, T3)]),\
          ]
  return feats

class ThreeVSTwoKeepawayExtractor(FeatureExtractor):
  def __init__(self):
    self.size = 1.0 # size of the domain
    self.tileNum = 40

    self.distMax = 1.415
    self.angleMax = 3.15

    self.distTileWidth = 0.15
    self.angleTileWidth = 0.2

    self.distTileOffset = (self.distMax - self.distTileWidth) / self.tileNum
    self.angleTileOffset = (self.angleMax - self.angleTileWidth) / self.tileNum

  def getFeatures(self, state, action):
    """
      Parse distances and angles to different objects
    """
    features = util.Counter()
    def setPositive(featureId, value, action, tileWidth, tileOffset):
      for i in range(int((value - tileWidth) / tileOffset), int(value / tileOffset)):
        features[(featureId, i, action)] = 1
      
    feats = threeVSTwoKeepawayFeatures(state, self.size)
    for i in xrange(11):
      setPositive(i, feats[i], action, self.distTileWidth, self.distTileOffset)
    for i in xrange(11, 13):
      setPositive(i, feats[i], action, self.angleTileWidth, self.angleTileOffset)
    
    return features

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

class GridExtractor(FeatureExtractor):
  """
  Extract the row number and column number
  """
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[0] = 1 # biase

    print state, action
    # position feature
    feats[1] = state[0]
    feats[2] = state[1]
    
    # offset to represent actions
    if action == 'north':
      feats[2] += 1
    elif action == 'south':
      feats[2] -= 1
    elif action == 'west':
      feats[1] -= 1
    elif action == 'east':
      feats[1] += 1
    
    return feats

class HorizontalExtractor(FeatureExtractor):
  """
  Extract the column number of a world
  """
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[0] = 1 # biase

    feats[1] = state[0]

    if action == 'west':
      feats[1] -= 0.25
    elif action == 'east':
      feats[1] += 0.25

    return feats

class BairdsExtractor(FeatureExtractor):
  """
    This class is for features extraction.
    THIS IS FOR BAIRD'S CONTEREXAMPLE GRID.
  """
  def getFeatures(self, state, action):
    feats = util.Counter()
    if state[1] == 1:
      # the first row states
      # Theta(6) + 2 * Theta(i)
      feats[6] = 1
      feats[state[0]] = 2
    else:
      # the state which can transit to terminal state
      # 2 * Theta(6) + Theta(i)
      feats[6] = 2
      feats[5] = 1
    return feats

def closestFood(pos, food, walls):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = util.Counter()
    
    features["bias"] = 1.0
    
    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    features.divideAll(10.0)
    return features
