# ECS170 Final Project
This is our final project for ECS 170 Spring 2018 at UC Davis

**Note to graders:** All logic has been implemented but a few bugs in the middle of the act/train step are preventing the entire model  from working. Environment and observation/reward modifiers are fully functional individually, it is just this training bug causing an issue.

## Project Goal
We wanted to build a bot to micromanage army units. Our model uses a deep
reinforcement learning algorithm called A2C and the FullyConv model to try to achieve this.
More generally, we wished to build a customizable system so that it can easily be adapted.
We achieve this customization through config.json files (or strings) that define what
observations, rewards, and actions the bot will consider while training. Similarly, all
advanced training settings, map files, and more can be set as well.

### References
We refered to these repos when considering designs for our project:
1. https://github.com/openai/baselines/blob/master/baselines/
2. https://github.com/simonmeister/pysc2-rl-agents/
3. https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
4. Other pysc2 git repos as well, but the above are the primary main ones

## Project Structure
`agent/`: general agent, model, and runner files (for A2C and FullyConv)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`modifiers/`: Modify and reformat observations, actions, and rewards  
`environment/`: environment manager  
`maps/`: map files  
`tests/`: test files (some unittest, some manual)  
`misc/`: images and other miscellaneous files  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`ObservationsResearch/`: Some experimenting with observations  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`WeeklyLogs/`: weekly check-in files  

>Some of our branches have more experimentation code we didn't want to clutter `master` with

### To run our code:
1. install pysc2 version 2.0 (from their git repo)
2. follow instructions on pysc2 website to install Starcraft 2 (latest)
3. install tensorflow gpu (1.8.0)
4. install numpy (1.14.2, should come with tensorflow)
5. run: `python3 main.py` (on 3.5+)

>Do note while all code is logically complete, there are a few unresolved bugs in the trainer

### Test Files

To run our test files, uncomment line 30 in main.py:

```py
    #test_env(env, config)
```

This tester shows that we are able to correctly set up the environment for
pysc2 and launch starcraft game instances. Here we are hard-coding actions,
but we are able to correctly parse observations and modify rewards. These can
be fed into the model.

In addition, actual unit tests can be found in the `tests/` folder as well.


### Our Map
We tested and trained on a simplified map. This map has no buildings, fog of war, or
resource collection. The tanks on this map are forced stationary and other units kept equal
between players same. For our project, we were interested in unit micromanagement and less
on other aspects of gameplay, so this map allows us to focus our model. It should be noted
that our AI can run any map file with a built in reward and even apply our own reward
calculations to any map.

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
