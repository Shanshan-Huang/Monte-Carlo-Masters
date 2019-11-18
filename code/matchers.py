__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import random
import pandas as pd
from abc import abstractmethod


class Matcher:
    def __init__(self):
        """
        Abstract matcher object. This object is used by the Market environment to match agent offers
        and also decide the deal price.
        """
        pass

    @abstractmethod
    def match(self,
              current_actions: dict,
              offers: pd.DataFrame,
              env_time: int,
              agents: pd.DataFrame,
              matched: set,
              done: dict,
              deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_actions: A dictionary of agent id and offer value
        :param offers: The dataframe containing the past offers from agents
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param matched: the set containing all the ids of matched agents in this round
        :param done: the dictionary with agent id as key and a boolean value to determine if an
        agent has terminated the episode
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the the agent id as keys and the rewards as values
        """
        rewards: dict = None
        return rewards


class RandomMatcher(Matcher):
    def __init__(self):
        """
        A random matcher, which decides the deal price of a matched pair by sampling a uniform
        distribution bounded in [seller_ask, buyer_bid] range.
        The reward is calculated as the difference from cost or the difference to budget for
        sellers and buyers.
        """
        super().__init__()

    def match(self,
              current_actions: dict,
              offers: pd.DataFrame,
              env_time: int,
              agents: pd.DataFrame,
              matched: set,
              done: dict,
              deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_actions: A dictionary of agent id and offer value
        :param offers: The dataframe containing the past offers from agents
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param matched: the set containing all the ids of matched agents in this round
        :param done: the dictionary with agent id as key and a boolean value to determine if an
        agent has terminated the episode
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the the agent id as keys and the rewards as values
        """
        # update offers
        for agent_id, offer in current_actions.items():
            if agent_id not in matched:
                offers.loc[offers['id'] == agent_id, ['offer', 'time']] = (offer, env_time)
        # keep buyer and seller offers with non-matched ids sorted:
        # descending by offer value for buyers
        # ascending by offer value for sellers
        # and do a second sorting on ascending time to break ties for both
        buyer_offers = offers[(offers['role'] == 'Buyer') &
                              (~offers['id'].isin(matched))] \
            .sort_values(['offer', 'time'], ascending=[False, True])

        seller_offers = offers[(offers['role'] == 'Seller') &
                               (~offers['id'].isin(matched))] \
            .sort_values(['offer', 'time'], ascending=[True, True])

        min_len = min(seller_offers.shape[0], buyer_offers.shape[0])
        rewards = dict((aid, 0) for aid in agents['id'].tolist())
        for i in range(min_len):
            considered_seller = seller_offers.iloc[i, :]
            considered_buyer = buyer_offers.iloc[i, :]
            if considered_buyer['offer'] >= considered_seller['offer']:
                # if seller price is lower or equal to buyer price
                # matching is performed
                print("buyer",considered_buyer['offer'] )
                print("seller",considered_seller['offer'] )
                matched.add(considered_buyer['id'])
                matched.add(considered_seller['id'])

                # keeping both done and matched is redundant
                done[considered_buyer['id']] = True
                done[considered_seller['id']] = True

                deal_price = random.uniform(considered_seller['offer'], considered_buyer[
                    'offer'])
                rewards[considered_buyer['id']] = considered_buyer['offer'] - deal_price
                rewards[considered_seller['id']] = deal_price - considered_seller['offer']
                matching = dict(Seller=considered_seller['id'], Buyer=considered_buyer['id'],
                                time=env_time, deal_price=deal_price)
                deal_history.append(matching)
            else:
                # not possible that new matches can occur after this failure due to sorting.
                break

        return rewards

class FirstPrice(Matcher):
    def __init__(self):
        """
        A Firstprice matcher, which sets the deal price to the first provided ask/bid. 
        Only offers that are not older than 10 seconds are considered (because  of draf.pdf section 2.3.2 page 6).
        It only considers values which are allowed w.r.t to the res_price (so initial conditions  are not considered if they are not allowed).
        If an agent provides the same offer as before it is not updated.
        After a deal was made the offer of the buyer is set to 0 and the offer of the seller is set to 100000.
        If there are multiple matches possible for an agent then the same choosing mechanism  is used as in Randommatch.
        (The agents are only compared sorted, biggest bid with smallest ask and so on)
        """
        super().__init__()
        
    def match(self,
              current_actions: dict,
              offers: pd.DataFrame,
              env_time: int,
              agents: pd.DataFrame,
              matched: set,
              done: dict,
              deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_actions: A dictionary of agent id and offer value
        :param offers: The dataframe containing the past offers from agents
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param matched: the set containing all the ids of matched agents in this round
        :param done: the dictionary with agent id as key and a boolean value to determine if an
        agent has terminated the episode
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the the agent id as keys and the rewards as values
        """
        # update offers
        for agent_id, offer in current_actions.items():
            if agent_id not in matched:
                #if the same offer as before is provide it is not updated
                if offer != offers.loc[offers['id']==agent_id,['offer']].iloc[0][0]:
                       offers.loc[offers['id'] == agent_id, ['offer', 'time']] = (offer, env_time) 
        # keep buyer and seller offers with non-matched ids sorted:
        # descending by offer value for buyers
        # ascending by offer value for sellers
        # and do a second sorting on ascending time to break ties for both
        # and check if the offers are valied w.r.t. res_price
        buyer_offers = offers[(offers['role'] == 'Buyer') &
                              (~offers['id'].isin(matched)) & (offers['offer'] <=offers['res_price'])&((offers['time']+10 )>=env_time)] \
            .sort_values(['offer', 'time'], ascending=[False, True])
        
        #
        seller_offers = offers[(offers['role'] == 'Seller') &
                               (~offers['id'].isin(matched)) & (offers['offer'] >=offers['res_price'])&((offers['time']+10 )>=env_time)] \
            .sort_values(['offer', 'time'], ascending=[True, True])
        display (buyer_offers)
        display(seller_offers)
        min_len = min(seller_offers.shape[0], buyer_offers.shape[0])
        rewards = dict((aid, 0) for aid in agents['id'].tolist())
        
        for i in range(min_len):
            considered_seller = seller_offers.iloc[i, :]
            considered_buyer = buyer_offers.iloc[i, :]
            if considered_buyer['offer'] >= considered_seller['offer']:
                # if seller price is lower or equal to buyer price
                # matching is performed
                matched.add(considered_buyer['id'])
                matched.add(considered_seller['id'])
                
                # keeping both done and matched is redundant
                done[considered_buyer['id']] = True
                done[considered_seller['id']] = True
                
                #check who provided the offer first for first price mechanism
                #if both provided at the same time then a coin flip (andom.randint(0, 1)) decides who was first
                if considered_buyer['time'] > considered_seller['time']:
                    deal_price = considered_seller['offer']
                elif considered_buyer['time'] < considered_seller['time']:
                    deal_price = considered_buyer['offer']    
                else:
                    if random.randint(0, 1) == 1:
                        deal_price = considered_seller['offer']
                    else:
                        deal_price = considered_buyer['offer']
                        
            
                rewards[considered_buyer['id']] = considered_buyer['offer'] - deal_price
                rewards[considered_seller['id']] = deal_price - considered_seller['offer']
                matching = dict(Seller=considered_seller['id'], Buyer=considered_buyer['id'],
                            time=env_time, deal_price=deal_price)
                deal_history.append(matching)
                #setting the offers to the values which shows thath this agent has made a deal
                offers.loc[offers['id'] == considered_buyer['id'], ['offer']] = (0)
                offers.loc[offers['id'] == considered_seller['id'], ['offer']] = (100000)
            else:
               # not possible that new matches can occur after this failure due to sorting.
               break

        return rewards        