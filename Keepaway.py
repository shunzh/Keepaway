# gridworld.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import mdp
import environment
import util
import optparse
import pickle
from sarsaLambdaAgents import SarsaLambdaAgent
from Crypto.Util.number import size

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
  def __init__(self, grid, size = 20, keeperNum = 3, takerNum = 2):
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
    if state == self.grid.terminalState:
      return ()
    x,y = state
    if type(self.grid[x][y]) == int:
      return ('exit',)
    return ('north','west','south','east')
    
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if state == self.grid.terminalState:
      return 0.0
    x, y = state
    cell = self.grid[x][y]
    if type(cell) == int or type(cell) == float:
      return cell
    return self.livingReward
        
  def getStartState(self):
    """
    Start state should be marked as 'S'
    If there are mutliple start states, choose randomly
    """
    startStateSet = []
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] == 'S':
          startStateSet.append((x, y))

    if startStateSet == []:
      raise 'Grid has no start state'
    else:
      return random.choice(startStateSet)
    
  def isTerminal(self, state):
    state[0]
                   
  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          
    """        
        
    if action not in self.getPossibleActions(state):
      raise "Illegal action!"
  

class KeepawayEnvironment(environment.Environment):
    
  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.gridWorld.getPossibleActions(state)
        
  def doAction(self, action):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.gridWorld.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.gridWorld.getStartState()

def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    display(state)
    pause()
    
    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    if len(actions) == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      return returns
    
    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'
    
    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")    
    # UPDATE LEARNER
    if 'observeTransition' in dir(agent): 
        agent.observeTransition(state, action, nextState, reward)
    
    returns += reward * totalDiscount
    totalDiscount *= discount

  if 'stopEpisode' in dir(agent):
    agent.stopEpisode()

 
if __name__ == '__main__':
  height, width = 20

  """
  opts.agent == 'sarsaApproximate'
  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': opts.discount, 
                'alpha': opts.learningRate, 
                'epsilon': opts.epsilon,
    'extractor': opts.extractor,
                'lambdaValue' : opts.lambdaValue,
                'replace' : opts.replace,
                'actionFn': actionFn}
  a = sarsaLambdaAgents.ApproximateSarsaAgent(**qLearnOpts)
  """
