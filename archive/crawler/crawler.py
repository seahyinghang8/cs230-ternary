#!/usr/bin/env python2
from downloader import download
import os
import urllib2

limit = 20

# download urls from imagenet
def imgNetDownload (url, name):
	# create directory
	os.mkdir(name)
	# read all img url from imagenet
	for img_url in urllib2.urlopen(url):
		# count the number of files in directory
		if(len(os.listdir(name)) > limit):
			print "Reached %d images. Time to move on" % limit
			break
		download(img_url.strip(), name)



# open names and labels file
names_file = open("names", "r")
labels_file = open("urllabels", "r")
base_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
base_dir = "datasets/"

# read the files and split it by newline
names = names_file.read().split("\n")
labels = labels_file.read().split("\n")

# close the input file
names_file.close()
labels_file.close()

# loop through all the names
counter = 0

for name in names:
	if (name == ""): continue

	folder_name = base_dir + str(name.strip())
	url = base_url + labels[counter]
	
	if (os.path.exists(folder_name)):
		print "%s already exist. Moving on to the next category" %(name)
		counter += 1
		continue
	
	print "Downloading: %s" %(name)
	imgNetDownload(url, folder_name)
	counter += 1
