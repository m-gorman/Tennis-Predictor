""" Evaluate performance of the system, based on profit and accuracy.
	Features used are those selected as the best for each player in
	montlyEval. """

from init import Unpkl, Player, Game, pklDump
import pandas as pd
import math
import itertools
import csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge, SGDClassifier
from sklearn.svm import SVC
from sklearn import svm, tree
import sys
from multiprocessing import pool
from sklearn.ensemble import AdaBoostClassifier
from utils import votingClassifier
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import gaussian_process

# initialize voting classifier
def initVC(vc):
	#vc.addClassifier(linear_model.LinearRegression())
	#vc.addClassifier(linear_model.Ridge (alpha = .5))
	#vc.addClassifier(linear_model.Lasso(alpha = 0.1))
	#vc.addClassifier(linear_model.LassoLars(alpha=.1))
	#vc.addClassifier(linear_model.BayesianRidge())
	#vc.addClassifier(svm.SVC())
	#vc.addClassifier(svm.SVC(decision_function_shape='ovo'))
	#vc.addClassifier(linear_model.SGDClassifier(loss="hinge", penalty="l2"))
	#vc.addClassifier(NearestNeighbors(n_neighbors=2, algorithm='ball_tree'))
	#vc.addClassifier(NearestCentroid())
	#vc.addClassifier(GaussianNB())
	#vc.addClassifier(tree.DecisionTreeClassifier())
	#vc.addClassifier(RandomForestClassifier(n_estimators=10))
	#vc.addClassifier(AdaBoostClassifier(n_estimators=100))
	#vc.addClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
	vc.addClassifier(linear_model.Ridge (alpha = .5))
	
def calcKellyBetSize(odds):
	p = 1 / (odds) + 0.02
	q = 1-p
	o = odds - 1
	
	betPercent = (o * p - q) / o

	return betPercent
	
	
games = Unpkl("GD.pkl")
playerData = Unpkl("PD.pkl")
playerFeatures = Unpkl("monthlyFeatures.pkl")

startDate = pd.to_datetime("1/1/2015",format='%d/%m/%Y')

count = 0
balance = 0
kBalance = 100
correct = 0
lstreak = 0
mbalance = 0
owed = 1
maxBet = 0
algo = svm.SVC
for g in games:
	if (g.data["Date"] >= startDate):
		gameMonth = g.data["Date"].month
		winner = g.data["Winner"]

		w_odds = g.data["wOdds"]
		l_odds = g.data["lOdds"]

		loser = g.data["Loser"]

		w_win = 'NaN'
		l_win = 'NaN'
		win_fail = 0
		
		
		# classify winner outcome
		w = playerFeatures[winner]

		thisGame = w.Games[ (w.Games["Date"] == g.data["Date"]) ]
		

		
		features = w.mf[gameMonth]["features"]
		
		# If no features were found, or those found had accuracy of less than 50%, ignore this game
		if (features == [] or w.mf[gameMonth]["acc"] < 50):
			#print "No features found for winner"
			continue
		#print features
		# get data
		GAMES = playerData[winner].getGamesUntil(g.data["Date"])
		Xs = GAMES[features]
		Y = GAMES["Won"]
		
		# get vector
		vec = thisGame[list(features)]
			
		#gnb = SGDClassifier(loss="hinge", penalty="l2")
		#gbc = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1, random_state=0)
		##supportvm = svm.SVC()
		#c = votingClassifier()
		#initVC(c)
		#c = linear_model.Ridge (alpha = .5)
		c = linear_model.Lasso(alpha = 0.1)
		#sgd = SGDClassifier()
		
		try:
			c.fit(Xs, Y)
			w_win = c.predict(vec)
		except:
			continue



		if (win_fail):
			continue
		# classify loser outcome
		w = playerFeatures[loser]
	
		thisGame = w.Games[ (w.Games["Date"] == g.data["Date"]) ]
		
		
		features = w.mf[gameMonth]["features"]
		# If no features were found, or those found had accuracy of less than 50%, ignore this game
		if (features == [] or w.mf[gameMonth]["acc"] < 50):
			continue
		#print "Loser features: ", features

		# get data
		GAMES = playerData[loser].getGamesUntil(g.data["Date"])
		Xs = GAMES[features]
		Y = GAMES["Won"]
		# get vector
		vec = thisGame[list(features)]
	
		#c = votingClassifier()
		#initVC(c)
		#c = linear_model.Ridge (alpha = .5)
		c = linear_model.Lasso(alpha = 0.1)
		#sgd = SGDClassifier()
		
		try:
			c.fit(Xs, Y)
			l_win = c.predict(vec)
		except:
			continue
	
	
		try:
			
			# system predicted winner to win
			if (w_win == 1 and  l_win == 0):#(w_win == 1 and l_win == 0 and w_odds > 2):
				print "Won"
				correct += 1
				count += 1
				balance += w_odds - 1
				
			# system predicted loser to win
			elif (w_win == 0 and l_win == 1):
				print "Lost"
				count += 1	
				lstreak += 1

			elif (w_win == l_win):
				print "No bet"
		except:
			pass
		print "%d / %d (%2.2f), balance: %f, kbalance: %d, maxBet: %f" % (correct, count, float(correct) / (count+1),  mbalance, kBalance, maxBet)
		
		
print "%d / %d, end balance: %f" % (correct, count, balance)





