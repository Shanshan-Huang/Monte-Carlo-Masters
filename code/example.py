import sys
sys.path.append('../')
# Load some generic libraries
import pandas as pd
import numpy as np
import warnings
# pandas setting warnings can be ignored, as it is intendend often
warnings.simplefilter("ignore")
from agents import Buyer, Seller
from environments import MarketEnvironment


#####################################################
# import pandas as pd
# data = pd.io.stata.read_stata('../../DataTimea')
# # use list(data) to show all the column names
# data.to_csv('../../DataTimeSeries.csv')
#####################################################


# Let's meet our agents
# The cost or budget for sellers and buyers, is also referred to as reservation price in general.
john = Seller('Seller John', 100)
nick = Seller('Seller Nick', 90)

sellers = [john, nick]


# Then buyers:
alex = Buyer('Buyer Alex', 130)
kevin = Buyer('Buyer Kevin', 110)

buyers = [alex, kevin]

# Now lets prepare our environment
# First let's load an information setting
# then let's load a matcher
from info_settings import BlackBoxSetting, DealFullInformationSetting, OtherSideSetting, SameSideSetting
from matchers import RandomMatcher


# Now let's create the environment
# market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=30,
#                                matcher=RandomMatcher(), setting=BlackBoxSetting)

market_env = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=30,
                               matcher=RandomMatcher(), setting=DealFullInformationSetting)

# Now let's run a single step, deciding offers for all agents:
# first we clean the environment, just in case
# everything should be zeroes.
init_observation = market_env.reset()


# Now for each agent we decide a price a bit above or lower from their cost or budget respectively for 2 steps.
# For step 1:
step1_offers = {
    'Buyer Alex': alex.reservation_price - 10.0,
 'Buyer Kevin': kevin.reservation_price - 5.0,
 'Seller John' : john.reservation_price + 10.0,
 'Seller Nick': nick.reservation_price +15.0
}
print(step1_offers)
observations, rewards, done, _ = market_env.step(step1_offers)
print('observations:')
print(observations)


# For step 2:

step2_offers = {
    'Buyer Kevin': kevin.reservation_price - 15.0,
    'Seller John' : john.reservation_price + 15.0,
}
print(step2_offers)
print('observations:')
observations, rewards, done, _ = market_env.step(step2_offers)
print(observations)



step3_offers = {
    'Buyer Alex': 20,
    'Seller Nick' : 100000,
}
observations, rewards, done, _ = market_env.step(step3_offers)


# now let's check when and if deals happened
pd.DataFrame(market_env.deal_history)

# and the history of offers
market_env.offers


# The above showcase a simple run of how two steps are done.
# After a whole round is finished, we call reset.
# In the lectures on 14.10.2019 and 21.10.2019, it will be shown how the MarketEnvironment is convertedto a gym environment.
# A single agent will be trained with reinforcement learning.
# Still, you are encouraged to expand the example yourselves and proceed with implementing the environment!


