# Agent-Based Modeling and Social System Simulation 2019

> * Group Name: Monte-Carlo Masters
> * Group participants names: 
      Guo Qifan,
      Huang Shanshan,
      Karbacher Till,
      Vidovic Gauzelin
> * Project Title: Reinforcement Learning for Markets

## General Introduction

This Project aims to simulate a double auction model with Reinforcement learning (RL). Out of the view of economic analysis it is interesting to find a competitive equilibrium. There are already some experiments done with humans for the double auction model, as for example from the *DeSciL* (Decision Making Laboratory of the ETHZ). RL seems promising for the simulation, since the aim of RL is to train software agents to decide between actions in an environment and maximize a cost function. 

## The Model

The double auction model consists of two types of players, buyers and sellers. Each have a price limit, also called valuation v, respectively their budget or their production price. At each timestep each player can submit a bid (buyers) or an ask (sellers). The bids have to be smaller than the valuation of the buyers (budget limit) and the asks have to be bigger than the valuation of the sellers (production coast).  If a buyer and a seller matches (buyers bid is higher than sellers ask) and a deal happens. The deal price is determined according to a matching mechanism. The bids and asks of the player depend on the information the players have. The amount of information is defined in the information settings. The aim of the players is to maximize the absolute value of the difference between their valuation v and deal price d.

We trained an agent in a model with one buyer and one seller. The random match mechanism and the full information settings was used.

## Fundamental Questions

**Will the trained agent be better than a random acting agent?**

If this question is fulfilled, then the agent was able to learn something and made a progress.

**Will the trained agent convert to an equilibrium?**

If this question is fulfilled, then the agent behaves in an optimal way. 

## Expected Results

Obviously, it was expected that an agent will be able to learn and preforms better than a random agent. 
It was expected that the convergency will be in a similar way as described in the draft.pdf of the *DeSciL* (see References).

## References 

[1]Timothy P. Lillicrap et al. “Continuous control with deep reinforcement learning”. In: (2015).
arXiv:1509.02971 [cs.LG]

[2]DeSciL.Feedback effects in the experimental double auction with private information - online
ressources.URL :https://osf.io/g84bt/ (Accessed: 06.12.19)

[3]Double auction wikpedia: URL: https://en.wikipedia.org/wiki/Double_auction (Accessed: 06.12.19)

## Research Methods

The Deep Deterministic Policy Gradient Algorithm was chosen (DDPG).  DDPG is a Reinforcement Learning algorithm for continuous action space (bids and asks are continuous). The continuous action space is a problem for a big part of the other RL algorithms.

The DDPG algorithm is:
 > * off-policy
> * model free
> * for environments with continuous action space
> * using an actor-critic approach
> * using a replay buffer (as DQN)

## Other

The first approach was to use the data set provide by the *DeSciL* but there were several inconsistencies in the data set. So, we did not use the dataset and made a self-learning agent.
Some of the inconsistencies of the dataset:
> * Players keep submitting bids or asks during rounds although they were already matched.
> * Matches did not occur even there would be possible buyers and sellers pairs. 
> * One side matches happened. Randomly some buyers/seller are matched, but no seller/buyer is matched.

## Code Reproducibility

Details on how to run the code is provided [here!](https://github.com/Shanshan-Huang/Monte-Carlo-Masters/tree/master/code)

