# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import math
import random
from featureExtractors import *

from learningAgents import ValueEstimationAgent
INF = 2**31-1 #float('inf')

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    
    for time in range(iterations):
      values = util.Counter()
      for state in mdp.getStates():
        if mdp.isTerminal(state): 
	  values[state] = 0
	else: 
          maxValue = -INF
	  for action in mdp.getPossibleActions(state):
	    maxValue = max(maxValue, self.getQValue(state, action))
	  values[state] = maxValue
      self.values = values
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    q = 0
    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
      q += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
    return q
    
  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    bestActions = []
    maxValue = -INF
    for action in self.mdp.getPossibleActions(state):
      q = self.getQValue(state, action)
      if q >= maxValue:
        maxValue = q
	bestActions.append (action)

    return random.choice (bestActions)

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
  def pickAction(self, state):
    "Returns the action according to probability distribution."
    return util.chooseFromDistribution()

  def getQValues(self):
    """
      Returns the Q counter - calculated asynchronously.
      Make sure this function is called for collecting data.
    """
    qValues = util.Counter()

    for state in self.mdp.getStates():
      for action in self.mdp.getPossibleActions(state):
        qValues[state, action] = self.getQValue(state, action)

    return qValues

class ApproximateValueIterAgent(ValueIterationAgent):
  """
     Similar to ApproximateQAgent.

     Using extractors, but only generalize on values of states,
     not state, action pairs.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    if extractor in args:
      extractor = args['extractor']

    self.featExtractor = util.lookup(extractor, globals())()

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.mdp = args['mdp']
    self.discount = args['gamma']
    self.iterations = args['iterations']
    self.alpha = args['alpha']

    self.weights = util.Counter()
    self.times = 0

    if False: #extractor == 'BairdsExtractor':
      # doing evil thing here
      self.weights[0] = 1
      self.weights[1] = 1
      self.weights[2] = 1
      self.weights[3] = 1
      self.weights[4] = 1
      self.weights[5] = 1
      self.weights[6] = 1
    
    # do update, full backup (sweep every state)
    for time in range(self.iterations):
      for state in self.mdp.getStates():
        if not self.mdp.isTerminal(state): 
	  # find the best action
          maxValue = None
	  bestAction = None

	  for action in self.mdp.getPossibleActions(state):
	    thisValue = self.getQValue(state, action)
	    if bestAction == None or thisValue > maxValue:
	      maxValue = thisValue
	      bestAction = action

          for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, bestAction):
	    self.update(state, action, nextState, self.mdp.getReward(state, action, nextState), prob)

      self.outputWeights(time)
      self.outputValues(time)
      self.outputMSE(time)

  """
  def getQValue(self, state, action):
    # Don't need to override getQvalue.
    # Make sure the Q values are calculated from values of states
    # which actually with function approximation applied
  """

  def getValue(self, state):
    """
      Should return V(state) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    v = 0.0

    # this feature should be designed for not caring about action.
    # passing None here. FIXME
    for feature, value in self.featExtractor.getFeatures(state, None).items():
      # weight * feature
      v += self.weights[feature] * value

    return v
    #util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward, prob = 1):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    correction = (reward + self.discount * self.getValue(nextState)) - self.getValue(state)
    for feature, value in self.featExtractor.getFeatures(state, None).items():
      self.weights[feature] += 1.0 / time * correction * value * prob
    #util.raiseNotDefined()

  def outputWeights(self, time):
    f = open("weights", "a")

    output = str(time) + ' '
    for weight in self.weights.values():
      output += str(weight) + ' '
    output += '\n'
    f.write(output)
    f.close()

  def outputValues(self, time):
    f = open("values", "a")

    values = [self.getValue(state) for state in self.mdp.getStates()]

    output = str(time) + ' '
    for value in values:
      output += str(value) + ' '
    output += '\n'
    f.write(output)
    f.close()

  def outputMSE(self, time):
    """
    for bairds problem only!
    """
    #FIXME

    values = [self.getValue(state) for state in self.mdp.getStates()]

    mean = 1.0 * sum(values) / len(values)

    error = math.sqrt(sum([(x - mean) ** 2 for x in values]))

    f = open("errors", "a")
    f.write(str(time) + ' ' + str(error) + '\n')
    f.close

  def final(self, state):
    "Called at the end of each game."
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
