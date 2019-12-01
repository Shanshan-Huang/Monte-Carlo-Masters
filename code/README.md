# Code Folder 

Dependencies:
numpy
panda
matplotlib
tensorflow
gym 
tf (1.14.0) would probably see some warninig, but it's okay
scipy
time

Sample commands to run the code is provided as following
```
python3 ddpg_combine_reward.py --reward_mode=1 --his_len=1 --noise=0
```
You can switch to different reward function by adjusting the `reward_mode`, set the number of historical length by setting  `his_len`, set `noise` level to a positive number to allow exploration. 

The reward function is plotted below, there are in total four different rewards to choose from.
![alt text](reward_func.png) "reward func")

It will output the accumulated reward as well as a picture in format `mode_x_hislen_x_var_x.png` with corresponding configuration decided in input argument.

