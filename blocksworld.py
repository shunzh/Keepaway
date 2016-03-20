import random
import sys
import mdp
import environment
import util
import optparse
import pickle

class Blocksworld(mdp.MarkovDecisionProcess):
  """
    Blocksworld

    Each state is a tuple of BlockTile
  """
  def __init__(self, count, stackNum):
    """
    count: number of blocks
    stacks: number of stacks that blocks can appear

    This is the initialization of the object.
    Initial states are defined in get_start_state.
    """
    self.count = count
    self.stackNum = stackNum

    self.states = []
    
    # parameters
    # These are assumed 0 if not set specifically.
    self.livingReward = 0
    self.noise = 0
        
  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise

  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    In form of [source, target], which moves a block from SOURCE to TARGET.

    Return empty action set if it's terminal state.
    """
    if self.isTerminal (state):
      return []

    state = decodeState(state)

    fromList = [i for i in range(self.stackNum) if state[i] != []]
    toList = range(self.stackNum)

    # we don't allow one block moves from a stack to itself - it's legal but doesn't make sense
    return [(source, target) for source in fromList for target in toList if source != target]
        
  def getStates(self):
    """
    Return list of all states.

    Here we only consider *observed* states, instead of the whole searching space
    """
    return self.states
            
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).

    Reward on reaching the target state.
    """
    if self.isTerminal(nextState):
      return 0
    else:
      return -1
        
  def getStartState(self):
    """
    Distribute blocks randomly

    FIXME It only appears in decreasing order in any stack from bottom to top
    """
    # a state is a list of stacks
    state = []
    for i in range(self.stackNum):
      state.append([])

    # select tiles to put in blocks
    for elem in range(self.count):
      stack_index = int (random.random() * self.stackNum)
      state[stack_index].append (elem)
      
    #print "start at: ", state

    return encodeState(state)
        
  def isTerminal(self, state):
    """
    A terminal state is: every blocks are on one stack, ordered.
    e.g. [3, 2, 1, 0] [] []
    """
    state = decodeState(state)

    for stack in state:
      if stack == []:
        continue
      else:
        # it's not empty, so all the blocks should be here
	if len(stack) != self.count:
	  return False

        # all blocks should be placed in order
	for index in range(self.count):
	  if not stack[index] == self.count - 1 - index:
	    return False
	
	return True

    # trivial case: no blocks
    return True
        
  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          

    Assume deterministic transition
    """
    source = action[0]
    target = action[1]

    # actually also copy original state
    nextState = decodeState(state)

    if nextState[source] == []:
      raise Exception("Source stack should not be empty.")
    else:
      elem = nextState[source].pop()

    nextState[target].append(elem)

    nextState = encodeState(nextState)

    self.states.append(nextState)
    return [(nextState, 1)]

class BlocksworldEnvironment(environment.Environment):
    
  def __init__(self, blocksworld):
    self.blocksworld = blocksworld
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.blocksworld.getPossibleActions(state)
        
  def doAction(self, action):
    successors = self.blocksworld.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.blocksworld.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.blocksworld.getStartState()

def encodeState(state):
  """
    Encode the list as tuple.
  """
  elements = [tuple(elem) for elem in state]
  return tuple(elements)

def decodeState(state):
  """
    Reverse of encodeState
  """
  elements = [list(elem) for elem in state]
  return list(elements)

def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).
  
  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action
def printString(x): print x

def runEpisode(agent, environment, discount, decision, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
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

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-x', '--lambda',action='store',
                         type='float',dest='lambdaValue',default=0.9,
                         help='Lambda for SARSA(lambda) (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-f', '--feature',action='store', metavar="A",
                         type='string',dest='extractor',default="IdentityExtractor",
                         help='Type of extractor if function approximation is applied.')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-y', '--replace',action='store_true',
                         dest='replace',default=False,
                         help='Replacing trace applied')
    optParser.add_option('-z', '--fileoutput',action='store_true',
                         dest='fileoutput',default=False,
                         help='File output')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()
    
    if opts.manual and opts.agent != 'q':
      print '## Disabling Agents in Manual Mode (-m) ##'
      opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:      
      opts.pause = False
      # opts.manual = False
      
    if opts.manual:
      opts.pause = True
      
    return opts

 
if __name__ == '__main__':
  
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  # 4 blocks and 3 stacks by default
  # FIXME pass parameters from command line
  mdp = Blocksworld(2, 2)
  env = BlocksworldEnvironment(mdp)
    
  ###########################
  # GET THE AGENT
  ###########################

  import valueIterationAgents, qlearningAgents, sarsaLambdaAgents, tglearningAgents
  a = None
  if opts.agent == 'value':
    a = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'valueApproximate':
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'iterations': opts.iters,
		  'mdp': mdp,
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
		  'extractor': opts.extractor,
                  'actionFn': actionFn}
    a = valueIterationAgents.ApproximateValueIterAgent(**qLearnOpts)
  elif opts.agent == 'q':
    #env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon
    #simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = qlearningAgents.QLearningAgent(**qLearnOpts)
  elif opts.agent == 'tg':
    #env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon
    #simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
		  'mdp': mdp,
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = tglearningAgents.TGLearningAgent(**qLearnOpts)
  elif opts.agent == 'qApproximate':
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
		  'extractor': opts.extractor,
                  'actionFn': actionFn}
    a = qlearningAgents.ApproximateQAgent(**qLearnOpts)
  elif opts.agent == 'sarsa':
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'lambdaValue' : opts.lambdaValue,
                  'replace' : opts.replace,
                  'actionFn': actionFn}
    a = sarsaLambdaAgents.SarsaLambdaAgent(**qLearnOpts)
  elif opts.agent == 'sarsaApproximate':
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
		  'extractor': opts.extractor,
                  'lambdaValue' : opts.lambdaValue,
                  'replace' : opts.replace,
                  'actionFn': actionFn}
    a = sarsaLambdaAgents.ApproximateSarsaAgent(**qLearnOpts)
  elif opts.agent == 'random':
    # # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 10
    class RandomAgent:
      def getAction(self, state):
        return random.choice(mdp.getPossibleActions(state))
      def getValue(self, state):
        return 0.0
      def getQValue(self, state, action):
        return 0.0
      def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
      def update(self, state, action, nextState, reward):
        pass      
    a = RandomAgent()
  else:
    if not opts.manual: raise 'Unknown agent type: '+opts.agent
    
    
  ###########################
  # RUN EPISODES
  ###########################
  messageCallback = lambda x: printString(x)
  if opts.quiet:
    messageCallback = lambda x: None

  # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
  pauseCallback = lambda : None
  if opts.pause:
    pauseCallback = lambda : display.pause()

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
  if opts.manual:
    decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
  else:
    decisionCallback = a.getAction  
    
  # RUN EPISODES
  if opts.episodes > 0:
    print
    print "RUNNING", opts.episodes, "EPISODES"
    print
  returns = 0

  #policyFile = open('policy' + opts.agent + str(opts.lambdaValue), 'a')
  #policyFile.write(str(opts.iters) + ' ' + str(checkPolicy(a)) + '\n')
  for episode in range(1, opts.episodes+1):
    thisReturn = runEpisode(a, env, opts.discount, decisionCallback, messageCallback, pauseCallback, episode)
    returns += thisReturn

  # debug: print out all the values learned
  #for state in mdp.getStates():
  #  print state, a.getValue(state)

  if opts.episodes > 0:
    print
    print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
    print
    print
    
  if (opts.agent == 'value') and not opts.manual:
    if opts.fileoutput:
       values = a.getValues(mdp.getStates())

       f = open("valueIterAnswer", "w")
       pickle.dump(values, f)


