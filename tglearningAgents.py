# TGlearningAgents.py
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
import sys
import numpy

class TGLearningAgent(ReinforcementAgent):
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
    self.predicate = BlocksworldPred(self.mdp.count, self.mdp.stackNum)
    self.root = QTreeNode(.01, self.mdp.count) # FIXME epsilon should passed via command line
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    nextState = self.mdp.getTransitionStatesAndProbs(state, action)[0][0] # FIXME know deterministic
    return self.root.lookup(self.getPred(nextState))
    
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
    actions = self.getLegalActions(state)
    max_v = None
    max_actions = []

    for action in actions:
      q = self.getQValue(state, action)
      if max_v == None or q > max_v:
        max_v = q
	max_actions = [action]
      elif q == max_v:
        max_actions.append(action)

    if max_actions == []:
      return None
    else:
      return random.choice(max_actions)
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

      Update the Q tree. Note that knowledge is in statistics and this specific
      update might not be seen in the next call of getQValue
    """
    "*** YOUR CODE HERE ***"
    sample = reward + self.gamma * self.getValue(nextState)
    new_qvalue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample 

    self.root.update(self.getPred(nextState), new_qvalue)

  def getPred(self, state):
    """
      Return the predicate that describes this state
    """
    ret = self.predicate.getPred(state)
    print state, ret
    return ret

class PredicateFactory ():
  def __init__ (self):
    pass

  def getPred (self, state):
    "Given a state representation, returns "
    raise Exception ("Need to be overriden.")

class BlocksworldPred (PredicateFactory):
  def __init__ (self, blockNum, stackNum):
    self.blockNum = blockNum
    self.stackNum = stackNum
    PredicateFactory.__init__(self)

  def getPred (self, state):
    pred = [] # predicates

    occupiedStk = 0
    # iterate each stack
    for stack in state:
      # iterate each block (except the one on the floor)
      # e.g. /*floor*/ (0, 1, 2)
      if stack != ():
        occupiedStk += 1
        for i in range(1, len(stack)):
          pred.append(('on', stack[i], stack[i - 1])) 
        pred.append(('on', stack[0], 'floor'))
        pred.append(('clear', stack[len(stack)-1]))

    # check whether possible to put block on the floor
    if occupiedStk < self.stackNum:
      pred.append(('clear', 'floor'))

    # return them
    ret = (tuple(pred)) # pity..just to make it hashable
    return ret

# a node on a Q-tree
class QTreeNode ():
  def __init__ (self, epsilon, blockNum):
    self.predicate = None
    self.stat = util.Counter()
    self.epsilon = epsilon # significance (not exploration!)
    self.possibleTests = [('on', i, j) for i in range(blockNum) for j in range(blockNum) + ['floor']]\
                       + [('clear', i) for i in range(blockNum) + ['floor']]

    print "QTreeNode: possible tests"
    print self.possibleTests

  def update (self, sample, value):
    if self.predicate == None:
      # leaf
      self.stat[sample] = value
      self.checkSplit()
    else:
      # go down the tree
      if self.predicate in sample:
        self.leftNode.update (sample, value)
      else:
        self.rightNode.update (sample, value)

  def lookup (self, sample):
    if self.predicate == None:
      # this is leaf
      if self.stat.values() == []:
        return 0
      else:
        # average of statistics
        return numpy.mean(self.stat.values())
    else:
      # go down the tree
      if self.predicate in sample:
        return self.leftNode.lookup (sample)
      else:
        return self.rightNode.lookup (sample)

  def checkSplit (self):
    if self.predicate != None:
      # only split the leaf
      raise Exception ("Cannot split a inner node.")
    # see every possible check
    for test in self.possibleTests:
      positive = [self.stat[sample] for sample in self.stat.keys() if test in sample]
      negative = [self.stat[sample] for sample in self.stat.keys() if not test in sample]

      splited = fTest (positive) + fTest (negative)
      joint = fTest (self.stat.values())

      print test, splited, joint
      print "positive:", positive
      print "negative:", negative

def fTest (data):
  """
    return the value for F-test
  """
  if data == []:
    # variance is INF
    return sys.maxint

  squaredSum = sum([i ** 2 for i in data])
  linearSum = sum(data)
  return squaredSum - 1.0 * (linearSum ** 2) / len(data)

class PacmanQAgent(TGLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    TGLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action
    
class ApproximateQAgent(TGLearningAgent):
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
    TGLearningAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
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
    "Called at the end of each game."
    # for output of Baird's counterexample
    f = open("weights", "a")

    output = str(self.times) + ' '
    for i in range(7):
      output += str(self.weights[i]) + ' '
    output += '\n'
    f.write(output)
    f.close()

    self.times += 1
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
