# gridworld.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp
from sarsaLambdaAgents import ApproximateSarsaAgent
import util
from qlearningAgents import ApproximateQAgent
import pprint
import pickle
import os
import sys
import featureExtractors
import random

class Keepaway(mdp.MarkovDecisionProcess):
  """
    Keepaway world
    
    State: (ball, keeper1 ... keeper3, taker1, taker2) 
    Action:
      hold, pass (if any keeper keeps the ball)
      None (o.w.)
    Transition:
      Keeper possessing the ball: depends on action
      Keeper not possessing the ball: 
        closest: go to the ball
        others: get open for a pass
      Taker: two takers run towards the ball, the other one tries to block passing lanes
  """
  def __init__(self, size = 1.0, keeperNum = 3, takerNum = 2):
    self.size = size
    self.keeperNum = keeperNum
    self.takerNum = takerNum
    
    self.ballSpeed = 0.025
    self.ballAttainDist = 0.03
    self.moveSpeed = 0.02
    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if self.weHaveBall(state):
      return [('hold',)] + [('pass', i) for i in range(1, self.keeperNum)]
    else:
      return [None]
    
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if self.isTerminal(nextState):
      return 0
    else:
      return 1
        
  def getStartState(self):
    size = self.size
    # need to specify init locations
    if self.keeperNum == 3 and self.takerNum == 2:
      return ((0.1, size - 0.1, 0, 0), (0, size), (size, 0), (size, size),\
                                       (0.1, 0), (0, 0.1))
    elif self.keeperNum == 4 and self.takerNum == 3:
      return ((0.1, size - 0.1, 0, 0), (0, size), (size, 0), (size, size), (size / 2, size / 2),\
                                       (0.1, 0), (0, 0.1), (0.1, 0.1))
    else:
      raise Exception("Unknown configuration.")
    
  def weHaveBall(self, state):
    ballLoc = state[0][:2]
    dist = util.getDistance(state[1], ballLoc)
    if dist < self.ballAttainDist:
      return True
    return False
  
  def opponentGetsTheBall(self, state):
    ballLoc = state[0][:2]
    for loc in self.getTakers(state):
      dist = util.getDistance(loc, ballLoc)
      if dist < self.ballAttainDist:
        return True
    return False

  def isTerminal(self, state):
    ballLoc = state[0]
    if ballLoc[0] < 0 or ballLoc[0] > self.size or ballLoc[1] < 0 or ballLoc[1] > self.size:
      #print "Out of playground"
      return True
    elif self.opponentGetsTheBall(state):
      #print "Opponent gets the ball"
      return True
    else:
      return False
                   
  def moveTowards(self, loc, dest):
    # move a step from loc to dest, and return the new loc
    diff = util.getDirection(loc, dest)
    diff = (diff[0] * self.moveSpeed, diff[1] * self.moveSpeed)
    return (loc[0] + diff[0], loc[1] + diff[1])

  def getLeastCongestedLoc(self, state, loc):
    def getCongestion(pos):
      congest = 0
      for i in range(1, self.keeperNum + self.takerNum + 1):
        if state[i] != loc:
          congest += 1.0 / (util.getDistance(pos, state[i]) + 0.0001)
      return congest
    
    buffer = 0.1
    mesh = 1.0 * (self.size - buffer * 2) / 4
    minCongest = 100
    minLoc = None
    for i in xrange(5):
      for j in xrange(5):
        x = buffer + mesh * i
        y = buffer + mesh * j
        congest = getCongestion((x, y))
        if congest < minCongest:
          minCongest = congest
          minLoc = (x, y)
    return minLoc
  
  def getKeepers(self, state):
    return state[1:self.keeperNum+1]
  
  def getTakers(self, state):
    return state[self.keeperNum+1:]

  def getTransitionStatesAndProbs(self, state, action = None):
    """
    The agent takes the action, and the world proceeds one time step.
    None action indicates no ball processor
    """
    # may move ball
    ball = state[0][:2]
    if action == None:
      ballVelocity = state[0][2:]
    else:
      ballVelocity = (0, 0)

    keepers = list(self.getKeepers(state))
    takers = list(self.getTakers(state))

    chasers = sorted(keepers, key=lambda keeper: util.getPointVectorDistance(keeper, ball, ballVelocity))
    # most closest agent, possess the ball, or go to the ball 
    if self.weHaveBall(state):
      # j has the ball, its transition depends on the action
      if action[0] == 'hold':
        pass
      elif action[0] == 'pass':
        # pass the ball to a teammate
        rand = util.randomVector(0.1)
        target = keepers[action[1]]
        diff = util.getDirection(keepers[0], (target[0] + rand[0], target[1] + rand[1]))
        ballVelocity = (self.ballSpeed * diff[0], self.ballSpeed * diff[1])
      else:
        raise Exception('Unknown action')
    else:
      # j should go to the ball
      chasers[0] = self.moveTowards(chasers[0], ball)

    # other agents get open for a pass
    for i in xrange(1, len(chasers)):
      # concretely, this agent goes to a least congested place
      chasers[i] = self.moveTowards(chasers[i], self.getLeastCongestedLoc(state, chasers[i]))
    keepers = sorted(chasers, key=lambda keeper: util.getDistance(keeper, ball))
    
    for i in xrange(2):
      takers[i] = self.moveTowards(takers[i], ball)
    for i in xrange(2, len(takers)):
      takers[i] = self.moveTowards(takers[i], keepers[1])
    takers = sorted(takers, key=lambda taker: util.getDistance(taker, keepers[0]))
    
    newBall = (ball[0] + ballVelocity[0], ball[1] + ballVelocity[1],\
               ballVelocity[0], ballVelocity[1])
    newState = [newBall] + keepers + takers
    return [(tuple(newState), 1)]
  
  def output(self, state):
    import matplotlib.pyplot as plt
    plt.clf()
    fig = plt.gcf()

    ball = plt.Circle(state[0][:2],.01,color='r')
    fig.gca().add_artist(ball)

    for loc in self.getKeepers(state):
      keeper = plt.Circle(loc,.02,color='b')
      fig.gca().add_artist(keeper)

    for loc in self.getTakers(state):
      taker = plt.Circle(loc,.02,color='g')
      fig.gca().add_artist(taker)

    fig.show()
    plt.pause(0.01)

    # pirnt out the locations of agents in the current state
    """
    print "Ball:", state[0]
    print "Keepers:", state[1 : self.keeperNum + 1]
    print "Takers:", state[self.keeperNum + 1 :]
    """
    #raw_input("Press Enter to continue...")

if __name__ == '__main__':
  size = 1.0
  episodes = 1000
  PLOT = False
  EXPLORE = True
  TYPE = "43"#"32"

  if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
      PLOT = True
      EXPLORE = False
    elif sys.argv[1] == 'check':
      PLOT = True
    else:
      TYPE = sys.argv[1]
      ID = sys.argv[2]

  if TYPE == "32":
    mdp = Keepaway(keeperNum=3, takerNum=2); alpha = 0.1 / 200; extractor = "ThreeVSTwoKeepawayExtractor"
  elif "43" in TYPE:
    mdp = Keepaway(keeperNum=4, takerNum=3); alpha = 0.1 / 300; extractor = "FourVSThreeKeepawayExtractor"
  else:
    raise Exception("Unknown type of task")

  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 1, 
                'alpha': alpha,
                'epsilon': 0.02 if EXPLORE else 0,
                'lambdaValue': 0,
                'extractor': extractor,
                'actionFn': actionFn}
  agent = ApproximateSarsaAgent(**qLearnOpts)

  if TYPE == "43t" or TYPE == "43i":
    weights = pickle.load(open( "weights" + TYPE + ".p", "rb" ))
    agent.weights = featureExtractors.keepwayWeightTranslation(weights)
    agent.workingWeights = agent.weights.copy()

  tList = []
  for _ in xrange(episodes):
    t = 0
    lastT = None
    prevState = None
    prevAction = None
    state = mdp.getStartState()

    while True:
      if mdp.isTerminal(state) or t > 1000:
        agent.update(prevState, prevAction, state, t - lastT)
        break
      
      if mdp.weHaveBall(state):
        if lastT != None:
          agent.update(prevState, prevAction, state, t - lastT)
          #print prevAction
        prevState = state
        lastT = t

      action = agent.getAction(state)
      if action != None: prevAction = action
      nextStateInfo = mdp.getTransitionStatesAndProbs(state, action)[0]
      nextState, prob = nextStateInfo
      state = nextState
    
      if PLOT: mdp.output(nextState); print "Action:", action
      t += 1

    #pprint.pprint(agent.weights)
    agent.final(state)
    tList.append(t)
    print _, t
    
    if (_ + 1) % 100 == 0:
      pickle.dump(tList, open( "time" + TYPE + ID + ".p", "wb" ))
      pickle.dump(agent.weights, open( "weights" + TYPE + ID + "_" + str(_) + ".p", "wb" ))