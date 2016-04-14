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
import featureExtractors

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
  def __init__(self, size = 5, keeperNum = 3, takerNum = 2):
    self.size = size
    self.keeperNum = keeperNum
    self.takerNum = takerNum
    
    self.ballSpeed = 0.4
    self.ballAttainDist = 0.3
    self.moveSpeed = 0.3
    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    return [('hold',)] + [('pass', i) for i in range(1, self.keeperNum + 1)]
    
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if not self.isTerminal(nextState):
      return 1
    else:
      return 0
        
  def getStartState(self):
    size = self.size
    # need to specify init locations
    if self.keeperNum == 3 and self.takerNum == 2:
      return ((0, size, 0, 0), (0, size), (size, 0), (size, size),\
                               (0, 0), (0, 1))
    
  def getBallPossessionAgent(self, state):
    ballLoc = state[0]
    for i in range(1, self.keeperNum + self.takerNum + 1):
      dist = (state[i][0] - ballLoc[0]) ** 2 + (state[i][1] - ballLoc[1]) ** 2
      if dist < self.ballAttainDist:
        return i 
    return None

  def isTerminal(self, state):
    ballLoc = state[0]
    if ballLoc[0] < 0 or ballLoc[0] > self.size or ballLoc[1] < 0 or ballLoc[1] > self.size:
      return True
    elif self.getBallPossessionAgent(state) > self.keeperNum:
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
    
    buffer = 0.15
    mesh = 1.0 * (self.size - buffer * 2) / 5
    minCongest  = 100
    minLoc = None
    for i in xrange(6):
      for j in xrange(6):
        x = buffer + mesh * i
        y = buffer + mesh * j
        congest = getCongestion((x, y))
        if congest < minCongest:
          minCongest = congest
          minLoc = (x, y)
    return minLoc

  def getTransitionStatesAndProbs(self, state, action = None):
    """
    The agent takes the action, and the world proceeds one time step.
    None action indicates no ball processor
    """
    newState = []

    # may move ball
    ball = state[:1]
    ballVelocity = state[2:]

    # move keepers, just close to the ball
    distIndices = util.sortByDistances(state[1: self.keeperNum + 1], ball)
    map(lambda _: _+1, distIndices)
    j = distIndices[0]
    if j == self.getBallPossessionAgent(state):
      # j has the ball, its transition depends on the action
      if action[0] == 'hold':
        newLoc = state[j]
      elif action[0] == 'pass':
        # pass the ball to a teammate
        k = distIndices[action[1]]
        diff = util.getDirection(state[j], state[k])
        ballVelocity = (self.ballSpeed * diff[0], self.ballSpeed * diff[1])
      else:
        raise Exception('Unknown action')
    else:
      # j should go to the ball
      newLoc = self.moveTowards(state[j], ball)
    newState.append(newLoc)

    for j in distIndices[1:]:
      # other agents get open for a pass
      # concretely, this agent goes to a least congested place
      newLoc = self.moveTowards(state[j], self.getLeastCongestedLoc(state, j))
    newState.append(newLoc)
    
    # move takers
    for j in range(self.keeperNum + 1, self.keeperNum + 3):
      newLoc = self.moveTowards(state[j], ball)
      newState.append(newLoc)
    for j in range(self.keeperNum + 3, self.keeperNum + self.takerNum + 1):
      # for other keepers, not implemented yet
      pass
    
    newBall = (ball[0] + ballVelocity[0], ball[1] + ballVelocity[1],\
               ballVelocity[0], ballVelocity[1])
    newState = [newBall] + newState
    return [(tuple(newState), 1)]
  
  def output(self, state):
    # pirnt out the locations of agents in the current state
    print "Ball:", state[0]
    print "Keepers:", state[1 : self.keeperNum + 1]
    print "Takers:", state[self.keeperNum + 1 :]

if __name__ == '__main__':
  size = 5

  mdp = Keepaway()
  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9, 
                'alpha': 0.5, 
                'epsilon': 0.1,
                'extractor': "ThreeVSTwoKeepawayExtractor",
                'actionFn': actionFn}
  agent = ApproximateSarsaAgent(**qLearnOpts)

  state = mdp.getStartState()
  while True:
    if mdp.isTerminal(state):
      break
    
    agentId = mdp.getBallPossessionAgent(state)
    if agentId == None:
      nextStateInfo = mdp.getTransitionStatesAndProbs(state)
      nextState, prob = nextStateInfo
      action = None
    else:
      action = agent.getAction(state)

      nextStateInfo = mdp.getTransitionStatesAndProbs(state, action)
      nextState, prob = nextStateInfo
      reward = mdp.getReward(state, action, nextState)
      
      agent.update(state, action, nextState, reward)
    
    mdp.output(nextState)
    print "Action:", action
    
    state = nextState