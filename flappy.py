import pygame
from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from keras.utils import to_categorical
import random
import statistics


def train_bird(load_weights=False, train=True):
    game = PLE(FlappyBird(), fps=30, display_screen=True)
    FLAP = 119
    
    agent = Agent(load_weights, train)
#     weights_filepath = 'weights/trained_weights.hdf5'
#     if agent.load_weights:
#         agent.network.load_weights(weights_filepath)
        
    games_played = 0
    total_score = 0
    
    # Training
    while games_played < agent.runs:
        total_score = 0
        # Play a total of 100 games before updating weights
        for i in range(100):
            score = 0
            game.init()
            while not game.game_over():
                # Greedy exploration
                old_state = agent.get_state(game)
                # print(old_state)
                if random.uniform(0, 1) < agent.epsilon:
                    final_action = to_categorical(randint(0, 1), num_classes=2)  # [1,0] SAU [0,1]
                else:
                    prediction = agent.network.predict(old_state.reshape((1,5)))
                    # print(prediction)
                    final_action = to_categorical(np.argmax(prediction[0]), num_classes=2)

                
                game.act(game.getActionSet()[np.argmax(final_action)])
                reward = 0
                if game.getActionSet()[np.argmax(final_action)] == FLAP:
                    reward = agent.get_reward_after_flap(game)
                else:
                    reward = agent.get_reward(game)
                score += reward
                new_state = agent.get_state(game)

                if agent.train:
                    agent.remember(old_state, final_action, reward, new_state, game.game_over())
                    
            #print()
            print(f'Score: {score}    Epsilon: {agent.epsilon}    Gamma: {agent.gamma}')
            total_score += score
            
        if agent.train:
            agent.replay_new(agent.memory, agent.batch_size)

        games_played += 1
        print(f'GAME {games_played}    Score: {total_score}')
        
        # Adjust epsilon for greedy exploration
        if not agent.train:
            agent.epsilon = 0.0
            agent.gamma = 0.9
        else:
            if agent.epsilon > 0.05:
                agent.epsilon = 1 - (games_played * agent.epsilon_decay)
            if agent.gamma <= 0.9:
                agent.gamma = games_played * agent.gamma_decay


if __name__ == '__main__':
    train_bird()  # play and train network
    # train_bird(True, False)  # run with trained weights