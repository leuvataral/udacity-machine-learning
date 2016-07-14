import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import itertools

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        self.alpha = 0.9
        self.gamma = 0.1
        self.q_values = {}
        self.value = 5
        self.q_table()
        #print self.q_values
        
        
    def q_table(self):
        directions = ['forward', 'left', 'right']
        lights = ['red', 'green']
        actions = [None, 'forward', 'left', 'right']
        oncoming = right = left = actions
        #print directions, lights, actions, oncoming, right, left
        for i, j, k, l, m, n in itertools.product(directions, lights, oncoming, right, left, actions):
                    self.q_values[((i, j, k, l, m), n)] = self.value

    def argmax(self, state):
        max_q = 0.0
        for i in self.q_values:
            if i[0] == state:
                if self.q_values[i] > max_q:

                    '''
                    print i[0], i[1]
                    print self.q_values[i]
                    '''
                    
                    max_q = self.q_values[i]
                    action = i[1]
                
        return max_q, action
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left']
        old_state = self.state
        
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right'])
        
        old_q_max, action = self.argmax(self.state)
        

        '''
        print "*"    
        print 'old_q_max: ', old_q_max, ' action: ', action
        print "*"
        '''
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        '''
        if reward < 0:
            print '***************'
            print 'NEGATIVE REWARD'
            print '***************'
        '''
        
        # TODO: Learn policy based on state, action, reward

        '''
        print "*"
        print 'old_state: ', old_state
        print 'action: ', action
        #print 'alpha: ', self.alpha
        #print 'gamma: ', self.gamma
        print 'reward: ', reward
        print 'old_q_max: ', old_q_max
        print "*"
        '''

        new_next_waypoint = self.planner.next_waypoint()
        new_inputs = self.env.sense(self)
        
        new_state = new_next_waypoint, new_inputs['light'], inputs['oncoming'], inputs['right'], inputs['left']
        new_q_max, new_action = self.argmax(new_state)

        '''
        print "*"
        print 'new_q_max: ', new_q_max, ' new_action: ', new_action
        print 'new_state: ', new_state
        print "*"
        '''
        #print 'old_state: ', old_state, 'new_state: ', new_state

        #print 'old q value: ', self.q_values[((old_state), action)]
        
        self.q_values[((old_state), action)] = ((1 - self.alpha) * old_q_max) + (self.alpha * (reward + (self.gamma * new_q_max)))
    
        #print 'updated q value: ', self.q_values[((old_state), action)]
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
