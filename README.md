# Reinforcement-Learning-Cat-and-Mouse

This is my implementation of ONE STEP and N STEP LOOK AHEAD using Q LEARNING

In the code I implement a Q-Table and use it to traverse through a maze. I utilize an epsilon greedy algorithm for my Agent to explore and exploit.

The q table is implemented as a dictionary where each possible state contains 4 values that correspond to the quality of moving up, down, left or right. 

The maze is implemented as a 2D array.

The AI agents are trained over a variable amount of episodes(currently set to 10,000). During training, the agents will choose to explore or exploit at each move, and continiously update their Q-Table after each move.

After the training is over, the agents will run through inference. A final showdown, as I like to think of it.
