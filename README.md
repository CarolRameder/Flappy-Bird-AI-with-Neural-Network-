# Flappy-Bird-Reinforcement-learning


The purpose of the project is understanding how an agent receive different awards,
depending on the action it decide to take during the learning process . 
For training the neural network I used Keras 

A few more details : 

  -training data is divided into minibatches 
  
  -the heurstic for game state gives rewards as the bird gets throw the gates and tries to reach the level of the new gate
  
  -the quality of an action into a certain state of the game is computed by summing it's reward and 
   the sum of the rewards as if the game is played perfectly from now on
