import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return {-1, 1}
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [(1, 0.1, 10), (-1, 0.9, -10)] if state == 0 else []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        totalValue, nextCard, deckCard = state
        successors = []
        if deckCard is None:
            return successors
        totalCard = sum(deckCard)
        if action == 'Take':
            if nextCard is not None:
                totalValue_new = totalValue + self.cardValues[nextCard]
                deckCard_new = list(deckCard)
                deckCard_new[nextCard] -= 1
                deckCard_new = tuple(deckCard_new)
                reward = 0
                if sum(deckCard_new) == 0 and totalValue_new <= self.threshold:
                    reward = totalValue_new
                if sum(deckCard_new) == 0 or totalValue_new > self.threshold:
                    deckCard_new = None
                successors += [((totalValue_new, None, deckCard_new), 1, reward)]
            else:   # nextCard == None
                for i in range(len(deckCard)):
                    if deckCard[i] == 0:
                        continue
                    deckCard_new = list(deckCard)
                    totalValue_new = totalValue + self.cardValues[i]
                    deckCard_new[i] -= 1
                    deckCard_new = tuple(deckCard_new)
                    reward = 0
                    if sum(deckCard_new) == 0 and totalValue_new <= self.threshold:
                        reward = totalValue_new
                    if sum(deckCard_new) == 0 or totalValue_new > self.threshold:
                        deckCard_new = None
                    successors += [((totalValue_new, None, deckCard_new), float(deckCard[i]) / totalCard, reward)]
            return successors
        elif action == 'Peek':
            if nextCard is None:
                for i in range(len(deckCard)):
                    if deckCard[i] == 0:
                        continue
                    successors += [((totalValue, i, deckCard), float(deckCard[i]) / totalCard, -self.peekCost)]
            return successors
        else:   # Quit
            return [((totalValue, None, None), 1, totalValue)]
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    mdp = BlackjackMDP([5, 21], 5, 20, 1)
    return mdp
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        max_Q = 0
        if newState is not None:
            max_Q = max(self.getQ(newState, newAction) for newAction in self.actions(newState))
        difference = reward + self.discount * max_Q - self.getQ(state, action)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += self.getStepSize() * difference * v
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Simulate Q-learning
def simulateQL(mdp):
    mdp.computeStates()
    QLAlgorithm = QLearningAlgorithm(mdp.actions, mdp.discount(), identityFeatureExtractor)
    util.simulate(mdp, QLAlgorithm, 30000)
    QLAlgorithm.explorationProb = 0
    stateAndAction = {}
    for state in mdp.states:
        stateAndAction[state] = QLAlgorithm.getAction(state)
    return stateAndAction

# Simulate value iteration
def simulateVI(mdp):
    VIAlgorithm = ValueIteration()
    VIAlgorithm.solve(mdp)
    return VIAlgorithm.pi

def compare(ql, vi):
    numDiff = 0
    for k, v in ql.items():
        if vi[k] != v:
            numDiff += 1
            # print 'state', k, ':', v, 'for QL,', vi[k], 'for VI'
    print numDiff, 'different actions between Q-learning and ValueIteration in all.'

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
ql = simulateQL(smallMDP)
vi = simulateVI(smallMDP)
print 'SmallMDP'
compare(ql, vi)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
ql = simulateQL(largeMDP)
vi = simulateVI(largeMDP)
print 'LargeMDP'
compare(ql, vi)

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    feature = [(('totalAndAction', total, action), 1)]
    if counts is not None:
        featureKey = map(lambda x: 1 if x != 0 else 0, counts)
        feature += [(tuple(['cardPresence'] + featureKey + [action]), 1)]
        for i in range(len(counts)):
            featureKey = ('cardAndAction', i, counts[i], action)
            feature += [(featureKey, 1)]
    return feature
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

piVI = simulateVI(originalMDP)
fixedRL = util.FixedRLAlgorithm(piVI)
rewards = util.simulate(newThresholdMDP, fixedRL)
print 'Rewards for value iteration in 10 trails:'
print rewards

newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
QLAlgorithm = QLearningAlgorithm(newThresholdMDP.actions, newThresholdMDP.discount(), identityFeatureExtractor)
rewards = util.simulate(newThresholdMDP, QLAlgorithm)
print 'Rewards for Q-learning in 10 trails:'
print rewards
