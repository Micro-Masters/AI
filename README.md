# ECS170 Final Project
This is our final project for ECS 170 Spring 2018 at UC Davis

## Project Goal
We wanted to build a bot to micromanage army units. Our model uses a deep
reinforcement learning algorithm called A2C to try to achieve this.

### References
We refered to these repos when considering designs for our project:
1. https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
2. https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/environment.py
3. https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
4. Other pysc2 git repos too, but above is the main one

### Our Map
We tested and trained on a simplified map. This map has no buildings, fog
of war, or resource collection. The tanks on this map are stationary. For
our project, we were interested in unit micromanagement and less on other
aspects of gameplay, so this map allows us to focus our model.

#### Zerg_44_36
Here are some screenshots of part of this map:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_44_36_2.png)
another screenshot:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_44_36.png)

#### Zerg_only_44_36
We also made a map where our agent and the built-in agent have the same
number of units, all zerglings:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_only_44_36.png)
another screenshot:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_only_44_36_2.png)

## To run our code:
1. install pysc2 version 2.0
2. follow instructions on pysc2 website to install Starcraft (latest)
3. install tensorflow (1.8.0)
4. install numpy (1.14.2)
5. run: `python3 main.py` (3.5.6)

## Test Files

To run our test files, uncomment line 30 in main.py:

```py
    #test_env(env, config)
```

This tester shows that we are able to correctly set up the environment for
pysc2 and launch starcraft game instances. Here we are hard-coding actions,
but we are able to correctly parse observations and modify rewards. Once our
A2C model is working, these will be fed into the model. 

