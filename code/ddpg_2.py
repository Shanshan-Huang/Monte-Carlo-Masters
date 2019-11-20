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


np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################
HIS_LEN=1
NUM_SELLER=1
NUM_BUYER=1
NUM_AGENT=2
MAX_EPISODES = 10
MAX_EP_STEPS = 500
LR_A = 0.0005   # learning rate for actor
LR_C = 0.0005     # learning rate for critic
GAMMA = 0.9     # reward discount
action_bound = 1
SCALE=100
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 100
BATCH_SIZE = 1

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement,name):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.name_=name
        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_+'/Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_+'/Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0.0001, 0.05)
            init_b = tf.constant_initializer(1)
            net = tf.layers.dense(s, 100, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.relu, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                actions=actions/SCALE
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
            #print(self.a)

            #print(a_grads)
        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            
            # self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

            # gvs = zip(self.policy_grads, self.e_params)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            gradients, _ = tf.clip_by_global_norm(self.policy_grads, 1.0)
            self.train_op = opt.apply_gradients(zip(gradients, self.e_params))

###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_,name):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.name_=name
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_+'/Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_+'/Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            # self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # capped_gvs = [(tf.clip_by_value(grad, -10000., 100000.), var) for grad, var in gvs]
            gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
            #print(self.q)
            #print(a)
            #print(self.a_grads)
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.5)
            init_b = tf.constant_initializer(1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        _,los,q_, gradienttt=self.sess.run([self.train_op,self.loss,self.gamma * self.q_, self.gradients], feed_dict={S: s, self.a: a, R: r, S_: s_})
        #print("loss",los)
        print('TTTTTT gradient',gradienttt)
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory

        self.data[index, :] = transition
        self.pointer += 1
    def print_transition(self):
        for i in range(self.capacity):
            print(self.data[i,:])
    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

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


# Robustness: 4 market structures.
# Control: Regular.
# 10 rounds, 10 buyers and 10 sellers.
# 3 Variations: (i) Asymmetric buy-side. (ii) Asymmetric sell-side. (iii) Long.
# (i) 10 rounds, 20 buyers and 10 sellers.
# (ii) 10 rounds, 10 buyers and 20 sellers.
# (iii) 50 rounds, 10 buyers and 10 sellers.

# the dratf.pdf have maximum 50 rounds in each episode
warnings.simplefilter("ignore")
env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=50,
                               matcher=MyRandomMatcher(), setting=FullInformationSetting)
#gym.make(ENV_NAME)
#env = env.unwrapped
#env.seed(1)

state_dim = HIS_LEN*(NUM_AGENT) + 1 # extra one for his own valuation price
action_dim = 1


# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()
# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
with tf.variable_scope('0'):
    actor_0 = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT,name='0')
    critic_0 = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor_0.a, actor_0.a_,name='0')
        #print(critic_0.a_grads)
    actor_0.add_grad_to_graph(critic_0.a_grads)
'''with tf.variable_scope('1'):
    actor_1 = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT,name='1')
    critic_1 = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor_1.a, actor_1.a_,name='1')
    actor_1.add_grad_to_graph(critic_1.a_grads)'''
#print("jellp")
sess.run(tf.global_variables_initializer())

M_0 = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
#M_1 = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 50/SCALE # control exploration
price_list_0=[]
price_list_1=[]
price_list_2=[]
time_list=[]
t=0
t1 = time.time()
for i in range(MAX_EPISODES):
    observation = env.reset()
    recent_history=[]
    #for agent in  observation.keys()[: NUM_SELLER]:
        #recent_history.append(observation[agent][-HIS_LEN:])
    #for m in range(NUM_AGENT * HIS_LEN):
        #recent_history.append(100*np.random.rand())
    for m in range(HIS_LEN):
        recent_history.append(40/SCALE)
        price_list_0.append(40/SCALE)
        price_list_2.append(40/SCALE)
        time_list.append(40/SCALE)
        t=t+1
    for m in range(HIS_LEN):
        recent_history.append((100-m/2)/SCALE)
        price_list_1.append((100-m/2)/SCALE)
        
        #time_list.append(t)
        #t=t+1

    #### TO-DO: might change alex nick
    buyer_s0 = np.array(recent_history + [alex.reservation_price])
    seller_s1 = np.array(recent_history + [nick.reservation_price])

    #for i in range(NUM_SELLER):
        #for j in range(HIS_LEN):

    ep_reward_0 = 0
    ep_reward_1 = 0

    for j in range(MAX_EP_STEPS):

        #if RENDER:
            #env.render()

        # Add exploration noise
        a = actor_0.choose_action(buyer_s0)
        print("a",a)
        price_list_2.append(a)
        #a=np.random.normal(a, var)
        #print(a)
        a = np.clip(np.random.normal(a,var), 0, 1)    # add randomness to action selection for exploration
        print("aa",a)
        step_offers = {}
        step_offers['0']=float(a)

        #n = actor_1.choose_action(seller_s1)
        #a=np.random.normal(a, var)
        #print(a)
        #print(n)
        #nn = np.clip(np.random.normal(n, var), nick.reservation_price/SCALE, 1)    # add randomness to action selection for exploration
        #print(nn)
        n=0.45
        nn=n
        step_offers['1']=float(nn)
        #print("n",n)
        #print("nn", nn)
        price_list_0.append(a)
        price_list_1.append(n)
       

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
            #print("al",alex.reservation_price)
            #print("a",a)
            print("float~~done",10*float(alex.reservation_price-a))
            #print(10*float(alex.reservation_price-a))
            M_0.store_transition(buyer_s0, a, 10*float(alex.reservation_price-a) , buyer_s0_)
            #M_1.store_transition(seller_s1, nn, r_1 , seller_s1_)
        else:
            #print('Buyer:', a,'Seller:', nn, 'diff:', a-nn )
            print("float~~fail", float(a-nn))
            M_0.store_transition(buyer_s0, a,  float(a-nn), buyer_s0_)
            #print(int(min(n-nick.reservation_price,-1)))
            #M_1.store_transition(s, n, 10*int(min(n-nick.reservation_price,-1)), s_)
            #M_1.store_transition(s, n, s.reshape((NUM_AGENT, -1))[1][-1]-nick.reservation_price, s_)
            #M_1.store_transition(seller_s1, nn, -float(nn-a), seller_s1_)
        if M_0.pointer > MEMORY_CAPACITY:
            var *= .99   # decay the action randomness
            b_M = M_0.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]
            print("state_old", b_s)
            print("state_new",b_s_)
            if done_:
                print(np.expand_dims([10*float(alex.reservation_price-a)],axis=0).shape)
                critic_0.learn(np.expand_dims(buyer_s0,axis=0), np.expand_dims(a,axis=0), np.expand_dims([10*float(alex.reservation_price-a)],axis=0), np.expand_dims(buyer_s0_,axis=0))
            else:
                print(np.expand_dims( [0],axis=0).shape)
                #critic_0.learn(np.expand_dims(buyer_s0,axis=0), np.expand_dims(a,axis=0), np.expand_dims(10*float(alex.reservation_price-a),axis=0), np.expand_dims(buyer_s0_,axis=0))
                critic_0.learn(np.expand_dims(buyer_s0,axis=0), np.expand_dims(a,axis=0), np.expand_dims( [0],axis=0), np.expand_dims(buyer_s0_,axis=0))
            actor_0.learn(np.expand_dims(buyer_s0,axis=0))
            #critic_0.learn(b_s, b_a, b_r, b_s_)
            #actor_0.learn(b_s)

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

        # if done_:
        #     print('Episode:', i, ' Reward_0: %i' % int(ep_reward_0),' Reward_1: %i' % int(ep_reward_1), 'Explore: %.2f' % var,)
        #     #print(j)
        #     # break
        # if j == MAX_EP_STEPS-1:
        #     print('Episode:', i, ' Reward_0: %i' % int(ep_reward_0),' Reward_1: %i' % int(ep_reward_1), 'Explore: %.2f' % var,)
        #     if ep_reward_0 > -300:
        #         RENDER = True
        #     # break
     
       #
    #M_0.print_transition()
print('Running time: ', time.time()-t1)
plt.axis([0, 5000, 0, 200/SCALE])
#plt.plot(x, y, color="r", linestyle="-", linewidth=0.5)
plt.scatter(time_list, price_list_0,color="r",label="buyer_noise", s=1)
plt.scatter(time_list, price_list_1,color="b",label="seller", s=1)
plt.scatter(time_list, price_list_2,color="g",label="buyer_true", s=1)
print(price_list_1)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.savefig("test.png", dpi=500)