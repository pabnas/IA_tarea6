import gym
import random
import numpy
import pandas
import math
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
"""Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials."""

class QLearningAgent:
    def __init__(self, actions, epsilon=0.9, gamma=0.90, alpha=0.5, min_alpha=0.1, min_epsilon=0.001,ada_divisor=25,**args):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.epsilon = epsilon # exploration probability
        self.actions = actions
        self.qs = {} # state table
        self.min_alpha = min_alpha  # learning rate
        self.min_epsilon = min_epsilon  # exploration rate
        self.ada_divisor = ada_divisor  # only for development purposes

    def getQValue(self, state, action):
        if not (state in self.qs) or not (action in self.qs[state]):
            return 0.0
        else:
            return self.qs[state][action]

    def return_dict(self):
        return self.qs

    def getLegalActions(self,state):
        return self.actions

    # def getAction(self, state):
    #     action = None
    #     if util.flipCoin(self.epsilon):
    #         legalActions = self.getLegalActions(state)
    #         action = random.choice(legalActions)
    #     else:
    #         action = self.computeActionFromQValues(state)
    #     return action

    def getAction(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        q = [self.getQValue(state, a) for a in legalActions]
        maxQ = max(q)

        # this is the trick.
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - 0.5 *mag for i in range(len(legalActions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(legalActions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        return legalActions[i]

    def update(self, state, action, nextState, reward):
        """
        Update q-value of the given state
        """
        if not (state in self.qs):
            self.qs[state] = {}
        if not (action in self.qs[state]):
            self.qs[state][action] = reward
        else:
            maxqnew = max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)])
            diff = reward + self.gamma * maxqnew - self.qs[state][action]
            newQ = self.qs[state][action] + self.alpha * diff
            self.qs[state][action] = newQ

        # print "(s, a, s', r) = [%3d (%3.1f, %3.1f), %d, %3d (%3.1f, %3.1f), %.1f]" % \
        #     (state, self.getQValue(state,0), self.getQValue(state, 1), action, \
        #      nextState, self.getQValue(nextState,0), self.getQValue(nextState, 1), \
        #      reward)

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]



# Number of states is huge so in order to simplify the situation
# we discretize the space to: 10 ** number_of_features
n_bins = 8
n_bins_angle = 10


cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]


agent = QLearningAgent(actions=range(env.action_space.n),alpha=0.5, gamma=0.90, epsilon=0.1)
episodios = 1000
reward_by_episode = numpy.zeros([episodios])
reward_by_episode_prom = numpy.zeros([episodios])
sum_promedio=0

for i_episode in range(episodios):
    state = env.reset()
    agent.alpha = agent.get_alpha(i_episode)
    agent.epsilon = agent.get_epsilon(i_episode)

    for t in range(200):

        if i_episode>(episodios-10):
            env.render()
        # choose an action
        stateId = build_state([to_bin(state[0], cart_position_bins), 
                               to_bin(state[1], cart_velocity_bins),
                               to_bin(state[2], pole_angle_bins),
                               to_bin(state[3], angle_rate_bins)])
        action = agent.getAction(stateId)

        # perform the action
        state, reward, done, info = env.step(action)
        nextStateId = build_state([to_bin(state[0], cart_position_bins), 
                                   to_bin(state[1], cart_velocity_bins),
                                   to_bin(state[2], pole_angle_bins),
                                   to_bin(state[3], angle_rate_bins)])
        if done == False:
            # update q-learning agent
            agent.update(stateId, action, nextStateId, reward)
        else:
            reward = -200
            agent.update(stateId, action, nextStateId, reward)
            sum_promedio = sum_promedio + t+1
            reward_by_episode[i_episode] = t+1
            reward_by_episode_prom[i_episode] = sum_promedio/(i_episode+1)
            #print("Episode " + str(i_episode) + " finished after " + str(t+1) + " timesteps")
            break

plt.figure(1)
plt.plot(reward_by_episode)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.title('Reward by episode')
plt.grid(True)
#plt.savefig("reward_by_episode,alpha = 0.5,gamma = 0.9,epsilon = 0.1.png")
plt.show()

plt.figure(2)
plt.plot(reward_by_episode_prom)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.title('Average reward by episode')
plt.grid(True)
#plt.savefig("reward_by_episode_prom,alpha = 0.5,gamma = 0.9,epsilon = 0.1.png")
plt.show()

