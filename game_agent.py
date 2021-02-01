from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
from operator import add
import collections

class Agent(object):
    def __init__(self, load_weights=False, train=True):
        self.reward = 0
        self.gamma = 0
        self.target = 1
        self.predict = 0
        self.runs = 150
        self.memory_size = 1000
        self.batch_size = 100
        self.learning_rate = 0.3  # modify later?
        self.epsilon = 1  # for greedy exploration
        self.epsilon_decay = 1/20
        self.gamma_decay = 1/20
        
        self.first_layer = 50
        self.second_layer = 300
        self.third_layer = 50
        
        self.short_memory = np.array([])
        self.memory = collections.deque(maxlen=self.memory_size)
        
        self.weights = 'weights/trained_weights.hdf5'
        self.load_weights = load_weights
        self.train = train
        
        self.network = self.create_network()
        
    
    def create_network(self):
        network = Sequential()
        network.add(Dense(activation='relu', input_dim=5, units=self.first_layer))  # input_dim may change if we change state rep
        network.add(Dense(activation='relu', units=self.second_layer))
        network.add(Dense(activation='relu', units=self.third_layer))
        network.add(Dense(activation='relu', units=2))
        
        opt = Adam(self.learning_rate)
        network.compile(loss='mse', optimizer=opt)
        
#         if self.load_weights:
#             network.load_weights(self.weights)
        
        return network
    
    
    def height_category(self, vertical_distance):
        if vertical_distance < 8:
            return 0
        elif vertical_distance < 20:
            return 1
        elif vertical_distance < 125:
            return 2
        elif vertical_distance < 250:
            return 3
        else:
            return 4


    def distance_category(self, horizontal_distance):
        if horizontal_distance < 8:
            return 0
        elif horizontal_distance < 20:
            return 1
        elif horizontal_distance < 125:
            return 2
        elif horizontal_distance < 250:
            return 3
        else:
            return 4
    
    
    def get_state(self, game):
        game_state = game.getGameState()
#         vert_dist = abs(game_state['next_pipe_bottom_y'] - game_state['player_y'])
#         horiz_dist = game_state['next_pipe_dist_to_player']
#         hc = self.height_category(vert_dist)
#         dc = self.distance_category(horiz_dist)
        return np.array([game_state['player_y'], game_state['next_pipe_top_y'], game_state['next_pipe_bottom_y'], 
                         game_state['next_next_pipe_dist_to_player'], game.game_over()])
    

    def get_reward(self, game):
        game_state = game.getGameState()
        bird_y = game_state['player_y']
        
        alive_points = 0
        limit_points = 0
        if game.getFrameNumber() % 10 == 0:
            alive_points = 5
        if bird_y < 20:
            limit_points += -5
        if bird_y > 450:
            limit_pointer += -10
        if game.game_over():
            return -30
        return alive_points + limit_points
    
    
    def get_reward_after_flap(self, game):
        game_state = game.getGameState()
        bird_y = game_state['player_y']
        top_pipe_y = game_state['next_pipe_top_y']
        bottom_pipe_y = game_state['next_pipe_bottom_y']
        dist_to_next_pipe = game_state['next_next_pipe_dist_to_player']
        
        alive_points = 0
        if game.getFrameNumber() % 10 == 0:
            alive_points = 5
        if top_pipe_y < bird_y < bottom_pipe_y and dist_to_next_pipe == 135 and not game.game_over():
            return alive_points + 25
        if game.game_over():
            return -30
        return alive_points
    
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))
        
    
    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        
        for state, action, reward, next_state, game_over in minibatch:
            target = reward
            if not game_over:
                # Q(s,a) = r(s,a,s') + gamma*maxQ(s',[a,a'])
                target = reward + self.gamma * np.amax(self.network.predict(next_state.reshape((1,5)))[0])
            target_f = self.network.predict(state.reshape((1,5)))
            # print(target_f)
            target_f[0][np.argmax(action)] = target
            # print(target_f)

            self.network.fit(np.array([state]), target_f, epochs=1, verbose=0)