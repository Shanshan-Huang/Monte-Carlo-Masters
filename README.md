# Agent-Based Modeling and Social System Simulation 2019

> * Group Name: Monte-Carlo Masters
> * Group participants names: 
      Guo Qifan,
      Huang Shanshan,
      Karbacher Till,
      Vidovic Gauzelin
> * Project Title: Reinforcement Learning for Markets

## General Introduction

(States your motivation clearly: why is it important / interesting to solve this problem?)
(Add real-world examples, if any)
(Put the problem into a historical context, from what does it originate? Are there already some proposed solutions?)

This Project aims to simulate a double auction model with Reinforcement learning (RL). Out of the view of economic analysis it is interesting to find a competitive equilibrium. There are already some experiments done with humans for the double auction model, as for example from the *DeSciL* (Decision Making Laboratory of the ETHZ). RL seems promising for the simulation, since the aim of RL is to train software agents to decide between actions in an environment and maximize a coast function. 

## The Model

(Define dependent and independent variables you want to study. Say how you want to measure them.) (Why is your model a good abtraction of the problem you want to study?) (Are you capturing all the relevant aspects of the problem?)

The double auction model consists of two types of players, buyers and sellers. Each have a price limit, also called valuation, respectively their budget or their production price. At each timestep each player can submit a bid (buyers) or an ask (sellers). The bids have to be smaller than the valuation of the buyers (budget limit) and the asks have to be bigger than the valuation of the sellers (production coast).  If a buyer and a seller matches (buyers bid is higher than sellers ask) and a deal happens. The deal price is determined according to a matching mechanism. The bids and asks of the player depend on the information the players have. The amount of information is defined in the information settings. 

We trained an agent in a model with one buyer and one seller. The random match mechanism and the full information settings was used

## Fundamental Questions

(At the end of the project you want to find the answer to these questions)
(Formulate a few, clear questions. Articulate them in sub-questions, from the more general to the more specific. )


## Expected Results

(What are the answers to the above questions that you expect to find before starting your research?)


## References 

(Add the bibliographic references you intend to use)
(Explain possible extension to the above models)
(Code / Projects Reports of the previous year)


## Research Methods

(Cellular Automata, Agent-Based Model, Continuous Modeling...) (If you are not sure here: 1. Consult your colleagues, 2. ask the teachers, 3. remember that you can change it afterwards)


## Other

(mention datasets you are going to use)
