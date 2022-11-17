# Flappy-Bird-Reinforcement-learning



The purpose of the project is to understand how an agent receives different awards, depending on the action it decides to take during the learning process. 
To build the neural network I used Keras. 

A few more details : 

  -training data is divided into minibatches 
  
  -the heuristic for game state gives rewards as the bird gets through the gates and tries to reach the level of the new gate
  
  -the quality of an action into a certain state of the game is computed by summing its reward and the sum of the rewards as if the game is played perfectly from now on
