""" Uses brute force to find the best features for each player.
	Performance is based on the percentage of games guessed correctly for a given
	feature set. Features for a given month are chosen based on the best performing
	subset of features in the preceding months. For example, the features used for a player
	in April 2015 will be those that performed best in the months Jan 2014 - March 2015. 
	
	Uses multiprocessing to utilise all cores. """

from init import Unpkl, Player, Game, pklDump
import pandas as pd
import math
import itertools
import csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import sys
from multiprocessing import pool
from multiprocessing.pool import ThreadPool as Pool

Allfeatures = ["Temp", "Hum", "rankDiff", "scoreDiff", "5Set", "Grass", "Hard", "Outdoor", "PP"]

# return all unique combinations of items in a list
def combinations(items):
	a = []
	for i in xrange(1,len(items)+1):
	    a += list(itertools.combinations(items,i))
	return a

featureCombos = combinations(Allfeatures)

games = Unpkl("GD.pkl")
playerData = Unpkl("PD.pkl")

startDate = pd.to_datetime("1/1/2015",format='%d/%m/%Y')

def votingClassifier(classifiers, Xs, Y, pred):
	net = 0
	for c in classifiers:
		classifier = c.fit(Xs, Y)
		if classifier.predict(pred)[0].astype(int) == 1:
			net += 1
		else:
			net -= 1
	if (net >= 0):
		return 1
	else:
		return 0

correct = 0
count = 0
balance = 10

# determine the effectiveness of a feature set
def evalFeatures(player, features, games, gameDate):
	correct = count = failed = balance = 0
	avgOdds = 0
	for game in games.iterrows():
		#try:
		data = game[1]
		date = data["Date"]

		# we find the best features on the 2014 season
		GAMES = player.getGamesUntil(date)

		Xs = GAMES[features]
		Y = GAMES["Won"]

		if Y.size <= 1:
			continue
		
		
		gnb =  GaussianNB()
		#gbc = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1, random_state=0)
		##supportvm = svm.SVC()
		#dt = tree.DecisionTreeClassifier()
		#sgd = SGDClassifier()

		classifiers = [gnb]#, gbc, supportvm, dt, sgd]


		vec = data[features]
		gnb.fit(Xs, Y)
		#vec.reshape(-1, 1)
		#win = votingClassifier(classifiers, Xs, Y, vec)
		win = gnb.predict(vec.reshape(-1, 1))[0].astype(int)

		#print win, data["Winner"]

		Odds = data["Odds"]

		if math.isnan(Odds):
			Odds = 1
		

		if win == 1 and data["Winner"] == player.name:
			correct += 1
			#print "won"
			balance += Odds - 1
			count += 1
			avgOdds += Odds

		elif win == 0 and data["Loser"] == player.name:
			count += 1
			correct += 1
		else:
			#print "lost"
			balance -= 1
		#print balance
			count += 1
	#except:
		#pass
		##print "EXCEPTION IN EVAL", sys.exc_info()[0]
		#print Xs
	#failed += 1

	return float(correct)/(1+count) * 100, 100 * float(failed) / (1 + count + failed), balance, count, avgOdds / (correct+1)

def chooseBest(player, date):
	games = player.getGames(pd.to_datetime("1/1/2014",format='%d/%m/%Y'), date)
	bestAcc = -1
	bestFeatures = []
	for featureSet in featureCombos:
		#print featureSet
		try:
			accuracy, failRate, balance, count, avgOddsCorrect = evalFeatures(player, list(featureSet), games, date)
			if accuracy > bestAcc:
				bestAcc = accuracy
				bestFR = failRate
				bestBalance = balance
				bestFeatures = featureSet
		except:
			pass
	return list(bestFeatures), bestAcc	

for player in playerData.keys():
	playerData[player].mf = {}

# worker function
def worker(month, playerData):
	monthStartDate = pd.to_datetime("1/" + str(month) + "/2015", format='%d/%m/%Y')
	print month
	numPlayers = len(playerData.keys())
	count = 0
	for player in playerData.keys():
		count += 1
		print month, count, numPlayers 
		bestFeatures, bestAcc = chooseBest(playerData[player], monthStartDate)
		playerData[player].mf[month] = {}
		playerData[player].mf[month]["features"] = bestFeatures
		playerData[player].mf[month]["acc"] = bestAcc
		print playerData[player].mf[month]
	
# amazon ec2 has 8 cores
pool_size = 8 
pool = Pool(pool_size)
	
# add each month's evaluation to the pool
for month in [1,2,3,4,5,6,7,8,9,10,11,12]:
	pool.apply_async(worker, (month, playerData))

pool.close()
pool.join()

pklDump(playerData, "monthlyFeatures.pkl")






