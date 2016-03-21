# gridworld.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp
from sarsaLambdaAgents import SarsaLambdaAgent, ApproximateSarsaAgent

class Keeper(SarsaLambdaAgent):
  def __init__(self):
    pass

class Taker():
  def __init__(self):
    pass

class Keepaway(mdp.MarkovDecisionProcess):
  """
    Keepaway world
  """
  def __init__(self, size = 20, keeperNum = 3, takerNum = 2):
    self.size = size
    self.keeperNum = keeperNum
    self.takerNum = takerNum
    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    return [('hold',), ('pass', 1), ('pass', 2), ('pass', 3)]
    
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
    return ((0, size, 0, 0), (0, size), (size, 0), (size, size),\
                             (0, 0), (0, 1))
    
  def getBallPossessionAgent(self, state):
    ballLoc = state[0]
    for i in range(1, self.keeperNum + self.takerNum + 1):
      dist = (state[i][0] - ballLoc[0]) ** 2 + (state[i][1] - ballLoc[1]) ** 2
      if dist < 0.5:
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
                   
  def getTransitionStatesAndProbs(self, state, action = None):
    """
    The agent takes the action, and the world proceeds one time step.
    None action indicates no ball processor
    """
    newState = []

    # move ball
    ballLoc = state[0]
    newState.append((ballLoc[0] + ballLoc[2], ballLoc[1] + ballLoc[3], ballLoc[2], ballLoc[3]))

    # move keepers, just close to the ball
    if action != None:
      i = self.getBallPossessionAgent(state)
    #TODO
    
    # move takers, depends on the action
    for j in range(self.keeperNum + 1, self.keeperNum + self.takerNum + 1):
      pass #TODO
    
    return [(newState, 1)]

if __name__ == '__main__':
  height, width = 20

  mdp = Keepaway()
  agent = ApproximateSarsaAgent()

  state = mdp.getStartState()
  while True:
    if mdp.isTerminal(state):
      break
    
    agentId = mdp.getBallPossessionAgent()
    if agentId == None:
      nextStateInfo = mdp.getTransitionStatesAndProbs(state)
      nextState, prob = nextStateInfo
    else:
      action = agent.getAction(state)

      nextStateInfo = mdp.getTransitionStatesAndProbs(state, action)
      nextState, prob = nextStateInfo
      reward = mdp.getReward(state, action, nextState)
      
      agent.update(state, action, nextState, reward)
    
    state = nextState