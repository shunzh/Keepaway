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
  def __init__(self, size = 1, keeperNum = 3, takerNum = 2):
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
    return [('hold',)] + [('pass', i) for i in range(1, self.keeperNum)]
    
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
                                       (0, 0), (0, 0.1))
    
  def getBallPossessionAgent(self, state):
    ballLoc = state[0][:2]
    for i in range(1, self.keeperNum + self.takerNum + 1):
      dist = util.getDistance(state[i], ballLoc)
      if dist < self.ballAttainDist:
        return i 
    return None
  
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

  def getLeastCongestedLoc(self, state, myId):
    def getCongestion(pos):
      congest = 0
      for i in range(1, self.keeperNum + self.takerNum + 1):
        if i != myId:
          congest += 1.0 / util.getDistance(pos, state[i])
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
    newState = []

    # may move ball
    ball = state[0][:2]
    if action == None:
      ballVelocity = state[0][2:]
    else:
      ballVelocity = (0, 0)

    distIndices = util.sortByDistToVector(self.getKeepers(state), ball, ballVelocity)
    distIndices = map(lambda _: _+1, distIndices)

    # most closest agent, possess the ball, or go to the ball 
    j = distIndices[0]
    if j == self.getBallPossessionAgent(state):
      # j has the ball, its transition depends on the action
      if action[0] == 'hold':
        pass
      elif action[0] == 'pass':
        # pass the ball to a teammate
        k = distIndices[action[1]]
        diff = util.getDirection(state[j], state[k])
        ballVelocity = (self.ballSpeed * diff[0], self.ballSpeed * diff[1])
      else:
        raise Exception('Unknown action')
      newLoc = state[j]
    else:
      # j should go to the ball
      newLoc = self.moveTowards(state[j], ball)
    newState.append(newLoc)

    # second closest agent, go to the ball 
    j = distIndices[1]
    newLoc = self.moveTowards(state[j], ball)
    newState.append(newLoc)

    # other agents get open for a pass
    for j in distIndices[2:]:
      # concretely, this agent goes to a least congested place
      newLoc = self.moveTowards(state[j], self.getLeastCongestedLoc(state, j))
      newState.append(newLoc)
    
    # move takers to the ball
    for loc in self.getTakers(state):
      newLoc = self.moveTowards(loc, ball)
      newState.append(newLoc)
    
    newBall = (ball[0] + ballVelocity[0], ball[1] + ballVelocity[1],\
               ballVelocity[0], ballVelocity[1])
    newState = [newBall] + newState
    return [(tuple(newState), 1)]
  
  def output(self, state):
    import matplotlib.pyplot as plt
    plt.clf()
    fig = plt.gcf()

    ball = plt.Circle(state[0][:2],.01,color='r')
    print ball
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
    print "Ball:", state[0]
    print "Keepers:", state[1 : self.keeperNum + 1]
    print "Takers:", state[self.keeperNum + 1 :]
    #raw_input("Press Enter to continue...")

if __name__ == '__main__':
  size = 1
  episodes = 5000

  mdp = Keepaway()
  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 1, 
                'alpha': 0.02,
                'epsilon': 0.2,
                'lambdaValue': 0,
                'extractor': "ThreeVSTwoKeepawayExtractor",
                'actionFn': actionFn}
  #agent = ApproximateSarsaAgent(**qLearnOpts)
  agent = ApproximateQAgent(**qLearnOpts)
  if os.path.exists('weights.p'):
    agent.weights = pickle.load(open( "weights.p", "rb" ))

  tList = []
  for _ in xrange(episodes):
    t = 0
    state = mdp.getStartState()
    while True:
      if mdp.isTerminal(state):
        break
      
      agentId = mdp.getBallPossessionAgent(state)
      if agentId == None:
        nextStateInfo = mdp.getTransitionStatesAndProbs(state, None)[0]
        nextState, prob = nextStateInfo
        action = None
      else:
        action = agent.getAction(state)

        nextStateInfo = mdp.getTransitionStatesAndProbs(state, action)[0]
        nextState, prob = nextStateInfo
        reward = mdp.getReward(state, action, nextState)
        
        agent.update(state, action, nextState, reward)
      
      #mdp.output(nextState)
      #print "Action:", action
      t += 1
    
      state = nextState

    if _ % 100 == 0:
      pickle.dump(tList, open( "time.p", "wb" ))
    #pprint.pprint(agent.weights)
    agent.final(state)
    print '#', _, t
    tList.append(t)

  pickle.dump(agent.weights, open( "weights.p", "wb" ))