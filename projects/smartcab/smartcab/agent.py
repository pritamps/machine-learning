import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np
from IPython.display import display


class QLearner:
    """
    General Q learner. Uses a tuple of (action, state) to maintain the history of Q-rewards
    """

    def __init__(self, actions=None, gamma=0.2, alpha=0.8, decay_epsilon=False, epsilon0=0.1):
        """
        Initializes the Q-learner
        :param actions: set of possible actions
        :param gamma: Learning rate
        """
        self.Q = {}
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.trial_number = 1
        self.epsilon0 = epsilon0
        self.epsilon = self.epsilon0
        self.t = 0
        self.decay_epsilon = decay_epsilon

    def get_q(self, state, action):
        """
        Gets Q-value given a state and action tuple. The default value is 1
        """
        return self.Q.get((state, action), 3)

    def update_q(self, current_state, reward, action, next_state):
        """
        Updates Q value based on the update equation.
        :param current_state: Current state
        :param next_state: State achieved by taking action 'action'
        :param action: Action taken at current state
        :param reward: Reward for taking action 'action' at state
        :return:
        """
        self.t += 1
        if (current_state, action) in self.Q:
            # The state has been visited before and needs updating
            val = self.Q[(current_state, action)]
            self.Q[(current_state, action)] = (1 - self.alpha) * val + self.alpha * (reward + self.gamma * self.get_max_a_q(next_state))
        else:
            # This is the first time visiting the state. Set the reward to the reward of the state
            self.Q[(current_state, action)] = 3

    def get_max_a_q(self, state):
        """
        :param state: State at which to calculate max_q
        :return: Maximum Q value possible from a current state
        """
        return max([self.get_q(state, action) for action in self.actions])

    def update_epsilon(self):
        """
        Updates the epsilon value to decay or to not decay
        :return:
        """
        if self.decay_epsilon:
            self.epsilon = self.epsilon0 / (self.t + 1)
        else:
            self.epsilon = self.epsilon0

    def get_best_action(self, state):
        """
        Returns best action given a state
        If more than one action produces the best value, a random choice among these actions is returned
        With a probability of epsilon (constant across a single trial), a random action is chosen
        :param state:
        :return:
        """
        self.update_epsilon()
        prob = random.random()

        if prob > self.epsilon:
            q_values = [self.get_q(state, action) for action in self.actions]
            max_q = max(q_values)
            n_max = q_values.count(max_q)
            print "Q_values: " + str(q_values)
            print "Max: " + str(q_values[q_values.index(max_q)])
            print "Actions: " + str(self.actions)
            print "Best action: " + str(self.actions[q_values.index(max_q)])
            candidate_actions = []
            if n_max > 1:
                for i in range(0, len(q_values)):
                    if max_q == q_values[i]:
                        candidate_actions.append(self.actions[i])
                return random.choice(candidate_actions)
            else:
                return self.actions[q_values.index(max_q)]
        else:
            return random.choice(self.actions)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, q=None):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.possible_actions = Environment.valid_actions
        self.Q = q

    def set_q_learner(self, q_learner):
        self.Q = q_learner

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.Q.trial_number += 1
        self.Q.update_epsilon()
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_state_tuple(self):
        inputs = self.env.sense(self)
        return inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.planner.next_waypoint()

    def update(self, t):
        current_state = self.get_state_tuple()
        self.next_waypoint = self.planner.next_waypoint()
        self.state = current_state

        action = self.Q.get_best_action(current_state)

        reward = self.env.act(self, action)
        print "Reward: " + str(reward)

        next_state = self.get_state_tuple()

        self.Q.update_q(current_state, reward, action, next_state)


def run_simulation(actions=None, gamma=None, alpha=None, epsilon0=None, decay_epsilon=None, live_plot=False):
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    # Set up Q-learner
    q_learner = QLearner(actions=Environment.valid_actions, gamma=gamma,
                         alpha=alpha, epsilon0=epsilon0, decay_epsilon=decay_epsilon)
    a.set_q_learner(q_learner)
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    # Live plot turned off to measure performance
    sim = Simulator(e, update_delay=0.01, display=False, live_plot=live_plot)
    sim.run(n_trials=100)  # run for a specified number of trials

    success_rate = float(sum(sim.rep.metrics['success'].ydata)) / len(sim.rep.metrics['success'].ydata)
    return success_rate


def run(perform_grid_search=False):
    """Run the agent for a finite number of trials."""

    # # Set up environment and agent
    # e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(LearningAgent)  # create agent
    # e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    if perform_grid_search:
        gammas = [0.2, 0.4, 0.5]
        alphas = [0.4, 0.6, 0.8]
        epsilons = [0.1, 0.3, 0.5]
        decay_epsilons = [True, False]

        numberOfRows = len(gammas) * len(alphas) * len(epsilons) * len(decay_epsilons)

        df = pd.DataFrame(index=np.arange(0, numberOfRows),
                          columns=['gamma', 'alpha', 'epsilon', 'decay_epsilon', 'success_rate'])
        index = 0
        for gamma in gammas:
            for alpha in alphas:
                for decay_epsilon in decay_epsilons:
                    for epsilon in epsilons:
                        success_rate = run_simulation(Environment.valid_actions, gamma=gamma,
                                                      alpha=alpha, epsilon0=epsilon, decay_epsilon=decay_epsilon)
                        row = [str(gamma), str(alpha), str(epsilon), str(decay_epsilon), "{0:.2f}".format(success_rate)]
                        df.loc[index] = row
                        index += 1

        df = df.sort(['success_rate', 'epsilon'], ascending=[0, 1])
        print display(df, index=False)
        print df.to_csv(index=False)
    else:
        success_rate = run_simulation(Environment.valid_actions, gamma=0.2,
                                      alpha=0.6, epsilon0=0.1, decay_epsilon=True, live_plot=False)
        print "Success rate: " + str(success_rate)
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run(perform_grid_search=True)
