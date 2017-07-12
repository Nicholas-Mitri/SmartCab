import random, numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import OrderedDict as OD

n_trials = 10000


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.state = None
        self.deadline = self.env.get_deadline(self)
        self.trip_len = []
        self.next_waypoint = random.choice(self.env.valid_actions[1:])

        self.qtable = dict()

        self.alpha = 0.6
        self.gamma = 0.4
        self.e = 0.1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.deadline = self.env.get_deadline(self)

    def update(self, t):
        # Gather inputs
        # self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state, current_state_str = self.get_agent_state()
        print "Agent State: {}\n".format(current_state_str)

        # TODO: Select action according to your policy
        if self.qtable.has_key(current_state_str):  # choose best action according to GLIE

            i = random.uniform(0, 1)
            if i >= self.e:
                action = np.argmax(self.qtable[current_state_str])  # argmax['forward', 'left', 'right']
                action = self.env.valid_actions[1:][action]
            else:
                action = random.choice(self.env.valid_actions[1:])
        else:
            self.qtable[current_state_str] = [0.0, 0.0, 0.0]
            action = random.choice(self.env.valid_actions[1:])

        # action = random.choice(Environment.valid_actions)  # choose random action
        # action = self.next_waypoint  # choose best action according to planner

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward >= 10:
            self.trip_len.append(self.deadline - deadline)

        # TODO: Learn policy based on state, action, reward
        new_state, new_state_str = self.get_agent_state()

        if not self.qtable.has_key(new_state_str):
            self.qtable[new_state_str] = [0.0, 0.0, 0.0]

        action_ind = self.env.valid_actions[1:] == action
        if self.qtable[current_state_str][action_ind] == 0.0:
            self.qtable[current_state_str][action_ind] = reward
        self.qtable[current_state_str][action_ind] = (1 - self.alpha) \
                                                     * (reward + self.gamma * np.max(self.qtable[new_state_str])) \
                                                     + self.alpha * self.qtable[current_state_str][action_ind]

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    action,
                                                                                                    reward)  # [debug]

    def get_agent_state(self):
        """returns the current state of the agent in an ordered dict and a string of that dict"""

        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        state = OD(inputs.items() +
                   [(x, y) for (x, y) in self.env.agent_states[self].items() if x in ['heading']] +
                   [('waypoint', self.next_waypoint)])

        return state, state.__str__()


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "\n\n--------------------------\n" \
          "Destination reached in {:0.2f} steps ON AVERAGE " \
          "with a STD of {:0.2f} steps and min/max of {:0.0f}/{:0.0f}" \
        .format(np.sum(a.trip_len) * 1.0 / len(a.trip_len), np.std(a.trip_len), np.min(a.trip_len), np.max(a.trip_len))
    print a.trip_len
    print "Success rate for this simulation is {:0.2f}% ({:0.0f}/{})\n" \
          "--------------------------".format(len(a.trip_len) * 100.0 / n_trials, len(a.trip_len), n_trials)

    print len(a.qtable)


if __name__ == '__main__':
    run()
