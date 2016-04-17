# sarsaLambdaAgents.py

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class SarsaLambdaAgent(ReinforcementAgent):
  """
    SarsaLearningAgent
    
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
    "initialize..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    # value counter
    self.values = util.Counter()
    self.workingValues = util.Counter()
    # initialize e_t
    self.e = util.Counter()
    # keeping the action of next state
    self.nextAction = None
  
  def getTargetQValue(self, state, action):
    return self.values[state, action]

  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.workingValues[state, action]
    
    #util.raiseNotDefined()
  
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
        q = self.getTargetQValue(state, action)
        if q > maxValue:
          maxValue = q
      maxActs = filter(lambda act: self.getTargetQValue(state, act) == maxValue, actions)
      return random.choice(maxActs)
    else:
      return None
    #util.raiseNotDefined()
    
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
    "*** YOUR CODE HERE ***"
    # when there it's now the TERMINAL_STATE
    if len(legalActions) == 0:
      return None

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
    delta = reward + self.gamma * self.getValue(nextState) - self.getValue(state)

    if self.replace:
      self.e[state, action] = 1
    else:
      self.e[state, action] += 1

    for state, action in self.workingValues:
      # here, update the target values
      self.workingValues[state, action] += self.alpha * delta * self.e[state, action]
      self.e[state, action] *= self.gamma * self.lambdaValue

  def final(self, state):
    # clear eligibility traces
    self.e = util.Counter()
    # copy current values to the target
    self.values = self.workingValues.copy()

class ApproximateSarsaAgent(SarsaLambdaAgent):
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
    SarsaLambdaAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    # we are not using values
    self.weights = util.Counter()
    self.workingWeights = util.Counter()
    self.times = 0
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    q = 0.0

    for feature, value in self.featExtractor.getFeatures(state, action).items():
      q += self.workingWeights[feature] * value
    return q
  
  def getTargetQValue(self, state, action):
    """
      Only used for getPolicy
    """
    q = 0.0

    for feature, value in self.featExtractor.getFeatures(state, action).items():
      q += self.weights[feature] * value
    return q
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    correction = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)

    featPairs = self.featExtractor.getFeatures(state, action).items()
    alpha = 0.1 / 200

    for feature, value in featPairs:
      self.e[feature] *= self.lambdaValue * self.gamma
      self.e[feature] += value
    
    for feature, value in self.featExtractor.getFeatures(state, action).items():
      self.workingWeights[feature] += alpha * correction * self.e[feature]

  def final(self, state):
    "Called at the end of each game."
    SarsaLambdaAgent.final(self, state)
    print util.getDictDistance(self.weights, self.workingWeights)
    self.weights = self.workingWeights.copy()
