""" Extract meaningful information from raw csv data to
	structured data that can be used by the rest of the application"""

# sort through game data and extract data for each player's game
import csv
import pandas as pd
import sys
import pickle

# columns that aren't needed
colsToDelete = ["Series", "W1", "W2", "W3", "W4", "W5", "L1", "L2", "L3", "L4", "L5", "Wsets", "Lsets", "Comment", "B365W", "B365L", "EXW", "EXL", "LBW", "LBL", "PSW", "PSL", "MaxW", "MaxL"]

# output file name
OUTPUTFILE = "PD.pkl"
GAMEFILE = "GD.pkl"

class Player:
    """Stores player's name and dataframe of their games"""
    def __init__(self, name):
        self.name = name
        self.bestFeatures = []


    def setFeatures(self, features):
    	self.bestFeatures = features

    # get games up until a certain date, non inclusive
    def getGamesUntil(self, date, startDate = pd.to_datetime("1/1/2010", format='%d/%m/%Y')):
    	return self.Games[ ( self.Games["Date"] >= startDate ) & ( self.Games["Date"] < date ) ]

    def getGames(self, startDate, endDate):
    	return self.Games[ ( self.Games["Date"] >= startDate ) & ( self.Games["Date"] < endDate ) ]

    # returns the feature vector (a list), to use as input in regression functions
	def buildFeatureVector(self, features):
		vec = []
		for feature in features:
			vec.append(self.data[feature])
		return vec

class Game:
	""" Stores information about a single tennis game, and methods to act on that information """
	def __init__(self, winner, loser, nSets, surface, court, wOdds, lOdds, date, wPoints, lPoints, wRank, lRank, temp, hum, tournament):
		self.data = {}
		self.data["Winner"] = winner
		self.data["Loser"] = loser
		if (nSets == 5):
			self.data["5Set"] = 1
		else:
			self.data["5Set"] = 0
		self.data["wOdds"] = wOdds
		self.data["lOdds"] = lOdds
		if (surface == "Grass"):
			self.data["Grass"] = 1
			self.data["Hard"] = 0
		elif (surface == "Clay"):
			self.data["Grass"] = 0
			self.data["Hard"] = 0
		elif (surface == "Hard"):
			self.data["Grass"] = 0
			self.data["Hard"] = 1
		if (surface == "Outdoor"):
			self.data["Outdoor"] = 1
		else:
			self.data["Outdoor"] = 0
		self.data["Date"] = date
		self.data["wPoints"] = wPoints
		self.data["lPoints"] = lPoints
		self.data["wRank"] = wRank
		self.data["lRank"] = lRank
		self.data["Temp"] = temp
		self.data["Hum"] = hum
		self.data["Tournament"] = tournament

	# returns the feature vector (a list), to use as input in regression functions
	def buildFeatureVector(self, features):
		vec = []
		for feature in features:
			vec.append(self.data[feature])
		return vec

	# calculate the rank difference
	# bool arg to see if it is for the loser (need to flip sign)
	def rankDiff(forLoser):
		sign = 1
		if forLoser:
			sign = -1
		return sign * (self.data["wRank"] - self.data["lRank"])


	# calculate the point difference
	# bool arg to see if it is for the loser (need to flip sign)
	def pointDiff(forLoser):
		sign = 1
		if forLoser:
			sign = -1
		return sign * (self.data["wPoints"] - self.data["lPoints"])

def readGames():
	frames = []
	for year in ["2010", "2011", "2012", "2013", "2014", "2015"]:
		frames.append( pd.read_csv(year + ".csv") )
	dfGames = pd.concat(frames, ignore_index=True)
	# fix up the dates so they are not just strings
	dfGames['Date'] = pd.to_datetime(dfGames.Date, format='%d/%m/%Y')
	# sort the games just in case
	dfGames = dfGames.sort('Date')
	dfGames.drop(colsToDelete, axis=1)
	return dfGames

# save item to pkl
def pklDump(item, fileName):
	pickle.dump(item, open(fileName, "wb"))

# unpickle a file
def Unpkl(fileName):
	return pickle.load( open( fileName, "rb" ) )

def init():
	games = readGames()
	gamesData = []

	for game in games.iterrows():
		winner = game[1]["Winner"]
		loser = game[1]["Loser"]
		wOdds = game[1]["AvgW"]
		lOdds = game[1]["AvgL"]
		nSets = game[1]["Best of"]
		surface = game[1]["Surface"]
		court = game[1]["Court"]
		date = game[1]["Date"]
		wRank = game[1]["WRank"]
		lRank = game[1]["LRank"]
		wPoints = game[1]["WPts"]
		lPoints = game[1]["LPts"]
		temp = game[1]["Temp"]
		hum = game[1]["Hum"]
		tournament = game[1]["Tournament"]
		newGame = Game(winner, loser, nSets, surface, court, wOdds, lOdds, date, wPoints, lPoints, wRank, lRank, temp, hum, tournament)
		gamesData.append(newGame)
	pklDump(gamesData, GAMEFILE)




	# get all unique players
	playerNames = set(games["Winner"].values.tolist() + games["Loser"].values.tolist())

	players = {}
	c = 0
	l = len(playerNames)
	# set up player objects
	for player in playerNames:
		print c, l
		c+=1

		newPlayer = Player(player)
		# get games player has played in
		playerGames = games[(games["Winner"] == player) | (games["Loser"] == player)]
		
		playerGames["Won"] = (playerGames["Winner"] == player).astype(int)
		
		# calculate rank difference. flip sign if player was loser
		playerGames["rankDiff"] = (playerGames["WRank"] - playerGames["LRank"]) * (((playerGames["Winner"] == player).astype(int)) - (playerGames["Loser"] == player).astype(int))
		
		# calculate score difference. flip sign if player was loser
		playerGames["scoreDiff"] = (playerGames["WPts"] - playerGames["LPts"]) * (((playerGames["Winner"] == player).astype(int)) - (playerGames["Loser"] == player).astype(int))
		
		# set court types
		playerGames["Grass"] = (playerGames["Surface"] == "Grass").astype(int)
		playerGames["Hard"] = (playerGames["Surface"] == "Hard").astype(int)
		#playerGames["Clay"] = (playerGames["Surface"] == "Clay").astype(int)
		playerGames["Outdoor"] = (playerGames["Court"] == "Outdoor").astype(int)
		
		# set num sets
		playerGames["5Set"] = (playerGames["Best of"] == 5).astype(int)
		
		# init past performance row
		playerGames["PP"] = 0
		
		
		
		# calculate past performance
		# + 1 for win, -1 for loss. count last 5 games
		for i, r in playerGames.iterrows():
			date = playerGames["Date"][i]
			pp = (playerGames["Won"][(playerGames["Date"] < date)].values.tolist()[-5:])
			sum = 0
			for game in pp:
				if game == 0:
					sum -= 1
				else:
					sum += 1
			playerGames.set_value(i, "PP", sum)
		
			winner = playerGames["Winner"][i]
			loser = playerGames["Loser"][i]
		
			if (player == winner):
				playerGames.set_value(i, "Opponent", loser)
			else:
				playerGames.set_value(i, "Opponent", winner)
		
		# set player odds
		playerGames["Odds"] = ((playerGames["Winner"] == player).astype(int) * playerGames["AvgW"]) + ((playerGames["Loser"] == player).astype(int) * playerGames["AvgL"])
		#playerGames["Odds"] = playerGames["Odds"] + ((playerGames["Loser"] == player).astype(int) * playerGames["AvgL"])
		
		playerGames.fillna(0)
		newPlayer.Games = playerGames
		
		players[player] = newPlayer
		
		# save all player info to disk
	pklDump(players, OUTPUTFILE)




if __name__ == "__main__":
	init()