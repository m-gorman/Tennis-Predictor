""" Contains functions to find the weather at a given location and time """
	
import urllib2
import json
import csv

# Create url that can be used to query information about a city from the google location api
def citifyUrl(city):
	return "https://maps.googleapis.com/maps/api/geocode/json?address=" + city + "&sensor=false&key=AIzaSyCBx28WPvSJtSnxESpsKhi10WYlo3QPVbc"

# get the latitude and longitude of a given city
def getLatLong(city):
	response = urllib2.urlopen(citifyUrl(city.replace(" ", "+")))
	data = json.load(response)
	locInfo = data["results"][0]["geometry"]["location"]
	lat = locInfo["lat"]
	long = locInfo["lng"]
	#print "%s at %s, %s" % (city, lat, long)
	return lat, long


# Find weather information of a given location at a given time
# Uses forecast.io
def getWeatherAtUnixTime(lat, long, time):
	url = "https://api.forecast.io/forecast/fcd67cdcda71c7a50aa411d5c0c70464/%s,%s,%s" % (lat, long, time)
#	print url
	response = urllib2.urlopen(url)
	data = json.load(response)
	temp = data["currently"]["temperature"]
	humidity = data["currently"]["humidity"]
	return temp, humidity
