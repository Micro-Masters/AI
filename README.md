# ECS170 Final Project
This is our final project for ECS 170 Spring 2018 at UC Davis

## Project Goal
We wanted to build a bot to micromanage army units. Our model uses a deep
reinforcement learning algorithm called A2C to try to achieve this.

### Our Map
We tested and trained on a simplified map. This map has no buildings, fog
of war, or resource collection. The tanks on this map are stationary. For
our project, we were interested in unit micromanagement and less on other
aspects of gameplay, so this map allows us to focus our model.

Here are some screenshots of part of this map:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_44_36_2.png)
another screenshot:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_44_36.png)
We also made a map where our agent and the built-in agent have the same
number of units, all zerglings:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_only_44_36.png)
another screenshot:
![alt text](https://github.com/Micro-Masters/AI/blob/master/misc/Zerg_only_44_36_2.png)

## To run our code:
1. install pysc2 version 2.0
2. follow instructions on pysc2 website to install Starcraft
3. install tensorflow
4. install pandas, numpy
5. run: python3 main.py
