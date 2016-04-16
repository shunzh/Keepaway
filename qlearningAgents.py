# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.values = util.Counter()
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.values[state, action]
   
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    bestAction = self.getPolicy(state)
    if bestAction: 
      return self.getQValue(state, bestAction)
    else: 
      return 0.0
   #util.raiseNotDefined()
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    maxValue = None
    actions = self.getLegalActions(state)
    if actions:
      for action in actions:
        q = self.getQValue(state, action)
        if q > maxValue:
          maxValue = q
      maxActs = filter(lambda act: self.getQValue(state, act) == maxValue, actions)
      return random.choice(maxActs)
    else:
      return None
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
   
    if util.flipCoin(self.epsilon): 
      action = random.choice(legalActions)
    else: 
      action = self.getPolicy(state)

    return action
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    sample = reward + self.gamma * self.getValue(nextState)
    new_qvalue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample 
    self.values[state, action] = new_qvalue 

class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    if extractor in args:
      extractor = args['extractor']

    self.featExtractor = util.lookup(extractor, globals())()
    QLearningAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
    self.oldWeights = util.Counter()
    self.times = 0

    if extractor == 'BairdsExtractor':
      # doing evil thing here
      self.weights[0] = 1
      self.weights[1] = 1
      self.weights[2] = 1
      self.weights[3] = 1
      self.weights[4] = 1
      self.weights[5] = 10
      self.weights[6] = 1
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    q = 0.0
    for feature, value in self.featExtractor.getFeatures(state, action).items():
      q += self.weights[feature] * value
    return q
    #util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    correction = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action)
    for feature, value in self.featExtractor.getFeatures(state, action).items():
      self.weights[feature] += self.alpha * correction * value
    #util.raiseNotDefined()

  def final(self, state):
    print util.getDictDistance(self.weights, self.oldWeights)
    self.oldWeights = self.weights.copy()