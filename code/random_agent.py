import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from agents import Buyer, Seller
from environments import MarketEnvironment
import tensorflow as tf
from info_settings import BlackBoxSetting,FullInformationSetting
from matchers import MyRandomMatcher
import gym
import time
import argparse
import scipy.stats as stats
parser = argparse.ArgumentParser()
parser.add_argument('--reward_mode', type=int, default=1, help='0:sparse reward; 1:discontinuous reward; 2:linear_peak; 3:squared function; 4:gaussian function; please see section 3.3 in the report for more details' )
parser.add_argument('--his_len', type=int, default=1, help='history length' )
parser.add_argument('--noise', type=int, default=0, help='noise' )
ARGS = parser.parse_args()
np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################
HIS_LEN=ARGS.his_len
NUM_SELLER=1
NUM_BUYER=1
NUM_AGENT=2
MAX_EPISODES = 1
MAX_EP_STEPS = 5000
SCALE=100

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'





 

def format_deal_history(observation):
    '''
    Takes in observation from the agent enviroment and returns a NUM_SELLER * HIS_LEN length numpy array.
    observation example
    {'Seller John': array([115. , 105.   , 120.   ,  95.  , 105.36013271,   0.        ]),
    'Seller Nick':  array([115.  , 105.   , 120.   ,  95.  , 105.36013271,   0.        ]),
    'Buyer Alex':  array([115.   , 105.   , 120.   ,  95.  , 105.36013271,   0.        ]),
    'Buyer Kevin': array([115.  , 105.   , 120.   ,  95.  , 105.36013271,   0.        ])}
    '''
    recent_history = []
    for agent in  observation.keys()[: NUM_SELLER]:
        recent_history.append(observation[agent][-HIS_LEN:])
    return np.array(recent_history)

alex = Buyer('0', 90/SCALE)
buyers = [alex]
sellers = []
'''for i in range(1, NUM_BUYER+1):
    buyers.append(Buyer(str(i), 120))
for i in range(NUM_BUYER+1, NUM_AGENT):
    sellers.append(Seller(str(i), 120))'''
nick = Seller('1', 30/SCALE)
sellers=[nick]



# the dratf.pdf have maximum 50 rounds in each episode
warnings.simplefilter("ignore")
env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=50,
                               matcher=MyRandomMatcher(), setting=FullInformationSetting)



var = 0/SCALE
#0/SCALE # control exploration
price_list_0=[]
price_list_1=[]
price_list_2=[]
time_list=[]
t=0
t1 = time.time()
ep_reward=0
for i in range(MAX_EPISODES):
    observation = env.reset()
    recent_history=[]
    #for agent in  observation.keys()[: NUM_SELLER]:
        #recent_history.append(observation[agent][-HIS_LEN:])
    for m in range(HIS_LEN):
        recent_history.append(40/SCALE)
        price_list_0.append(40/SCALE)
        price_list_2.append(40/SCALE)
        time_list.append(40/SCALE)
        t=t+1
    for m in range(HIS_LEN):
        recent_history.append((100-m/2)/SCALE)
        price_list_1.append((100-m/2)/SCALE)
        
    buyer_s0 = np.array(recent_history + [alex.reservation_price])
    seller_s1 = np.array(recent_history + [nick.reservation_price])

    ep_reward_0 = 0
    ep_reward_1 = 0

    for j in range(MAX_EP_STEPS):

        a=np.random.rand()
        print("a",a)
        price_list_2.append(float(a))
        print("aa",a)
        step_offers = {}
        step_offers['0']=float(a)

        n=0.45
        nn=n
        step_offers['1']=float(nn)

        price_list_0.append(float(a))
        price_list_1.append(float(n))
       

        time_list.append(t)
        t=t+1
        #alex.reservation_price-np.random.rand()*alex.reservation_price

        #####################################################################
        # load data. once the agent succeeded in the previous time step,
        # since the sellers and buyers might quit the markets at anytime, so the state space is actually varying in length
        # to keep the state space a fixed length variable, we decide to
        # increase the offer price to large price 10000 for Sellers and decrease the offer prcie to 0 for Buyers
        # once they are matched
        #
        #0 is agent 1-9 other buyer  10-19 seller

        curr_offer = np.zeros((NUM_AGENT,1))
        for buyer in buyers:
            curr_offer[int(buyer.agent_id)] = step_offers[buyer.agent_id]
        for seller in sellers:
            curr_offer[int(seller.agent_id)] = step_offers[seller.agent_id]

        '''for buyer in buyers:
            # print("buyer", buyer.agent_id)
            if (buyer.agent_id != '0'):
                step_offers[buyer.agent_id]=buyer.reservation_price+np.random.rand()*buyer.reservation_price
                curr_offer[int(buyer.agent_id)] = step_offers[buyer.agent_id]
            

        for seller in sellers:
            # print("seller", seller.agent_id)
            if (seller.agent_id != '0'):
                step_offers[seller.agent_id]=seller.reservation_price+np.random.rand()*seller.reservation_price
                curr_offer[int(seller.agent_id)] = step_offers[seller.agent_id]'''

        #####################################################################

        observation, rewards, done, _ = env.step(step_offers)
        print(step_offers)
        print("=============rewards", rewards)
        s_ = np.hstack((buyer_s0[:-1].reshape((NUM_AGENT, -1))[:, 1:], curr_offer)).flatten()
        #print("----s",s_)
        buyer_s0_ = np.hstack((s_, alex.reservation_price))
        seller_s1_= np.hstack((s_, nick.reservation_price))

        r_0 = rewards['0']
        r_1=rewards['1']
        print(done)
        done_=True
        for boolen in done.values():
            if not boolen:
                done_=False
        

        if done_:
            ep_reward+=float(alex.reservation_price-a)
            #M_1.store_transition(seller_s1, nn, r_1 , seller_s1_)
       
       

        '''if M_1.pointer > MEMORY_CAPACITY:
            var *= .9995   # decay the action randomness
            b_M = M_1.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]
            #print('reward',b_r)
            critic_1.learn(b_s, b_a, b_r, b_s_)
            actor_1.learn(b_s)'''

        buyer_s0 = buyer_s0_
        seller_s1 = seller_s1_

        ep_reward_0 += r_0
        ep_reward_1 += r_1
    
        print('Episode:', i, ' Reward_0: %i' % int(ep_reward_0), ' Reward_1: %i' % int(ep_reward_1), 'Explore: %.2f' % var,)

print('Running time: ', time.time()-t1)
print("hislength",HIS_LEN)
print("reward",ep_reward)
plt.axis([0, MAX_EPISODES*MAX_EP_STEPS+HIS_LEN*MAX_EPISODES, -1, 200/SCALE])
#plt.plot(x, y, color="r", linestyle="-", linewidth=0.5)
plt.scatter(range(len(price_list_1)), price_list_0,color="r",label="buyer_noise", s=1)
plt.scatter(range(len(price_list_1)), price_list_1,color="b",label="seller", s=3)
plt.scatter(range(len(price_list_1)), price_list_2,color="g",label="buyer_true", s=1)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.savefig("result/random.png" ,dpi=500)

# plt.axis([0, 200, 0, 1.1])
# plt.scatter(range(200), price_list_0[0:200],color="g",label="buyer_true", s=1)
# plt.scatter(range(200), price_list_2[0:200],color="r",label="buyer_true", s=1)
# plt.scatter(range(200), price_list_1[0:200],color="b",label="buyer_true", s=1)
# plt.savefig("bug.png", dpi=500)
