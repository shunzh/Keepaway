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
from util import getDistance

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

def threeVSTwoKeepawayFeatures(state, size):
  center = (size / 2, size / 2) #OVERFIT
  ball = state[0][:2]
  
  keepersId = util.sortByDistances(state[1:4], ball)
  takersId = util.sortByDistances(state[4:], ball)
  keepersId = map(lambda _: _+1, keepersId)
  takersId = map(lambda _: _+4, takersId)

  # get features as a list of real numbers
  dists = [getDistance(state[keepersId[0]], ball),\
           getDistance(state[keepersId[0]], state[keepersId[1]]),\
           getDistance(state[keepersId[0]], state[keepersId[2]]),\

           getDistance(state[keepersId[0]], state[takersId[0]]),\
           getDistance(state[keepersId[0]], state[takersId[1]]),\
           getDistance(state[keepersId[1]], center),\
           getDistance(state[keepersId[2]], center),\

           getDistance(state[takersId[0]], center),\
           getDistance(state[takersId[1]], center),\

           min(getDistance(state[keepersId[1]], state[takersId[0]]), getDistance(state[keepersId[1]], state[takersId[1]])),\
           min(getDistance(state[keepersId[2]], state[takersId[0]]), getDistance(state[keepersId[2]], state[takersId[1]])),\
           # and more
          ]

  return dists

class ThreeVSTwoKeepawayExtractor(FeatureExtractor):
  def __init__(self):
    self.size = 5.0
    self.tileNum = 32

  def getFeatures(self, state, action):
    """
      Parse distances and angles to different objects
    """
    features = [0] * 13 * self.tileNum
    def setPositive(featureId, value, tileSize):
      features[featureId * self.tileNum + value / tileSize] = 1
      
    distTile = None
    angleTile = None
    
    # try it first
    feats = threeVSTwoKeepawayFeatures(state, self.size)
    return {id: feats[id] for id in xrange(len(feats))}

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
