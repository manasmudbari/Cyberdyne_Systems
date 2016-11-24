# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 13:24:37 2016

@author: manasmudbari
"""

import pandas as pd
import numpy as np
import pickle
from urllib2 import Request, urlopen
import json
import sys

#this piece of code handles the ascii data
reload(sys)
sys.setdefaultencoding('utf-8')



repo_URL = 'https://api.github.com/repos/iron/router/stargazers?access_token=be6503a79d82e51f59646627acba4b321c3de8bd&page=5'
starGaze = pd.read_json(repo_URL)
##----Reshaping the DataFrame into a matrix
#user_str = starGaze['login'].as_matrix()
###----Searilizing and storing the API data locally
#user_out = open('user.pickle','ab')
#pickle.dump(user_str, user_out)
#user_out.close()

starGaze['login'].to_csv('github.csv', mode='a', header=False)