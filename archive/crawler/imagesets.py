import re
import urllib, urllib2

urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:42.0) Gecko/20100101 Firefox/42.0'}

#TODO: Create classes

def fetch_images_from_Bing(keyword, use_safe_search=True, maxResults=10000, bingcount=35):
	current = 1
	last = ''
	adlt = 'off' if use_safe_search == False else ''

	while True:
		request_url = 'https://www.bing.com/images/async?q=' + urllib.quote_plus(keyword) + '&async=content&first=' + str(current) + '&adlt=' + adlt
		request = urllib2.Request(request_url, None, headers=urlopenheader)
		response = urllib2.urlopen(request, timeout=5)
		html = response.read().decode('utf8')
		links = re.findall('imgurl:&quot;(.*?)&quot;',html)

		try:
			if links[-1] == last:
				break
			last = links[-1]
			current += bingcount
			for link in links:
				yield link

		except IndexError:
			print('No search results for "{0}"'.format(keyword))
			return

		if current >= maxResults:
			break

def fetch_images_from_ImageNet(keyword):
	request_url = 'http://image-net.org/search?q=' + keyword
	request = urllib2.Request(request_url, None, headers=urlopenheader)
	response = urllib2.urlopen(request, timeout=5)
	html = response.read().decode('utf8')
	sysnetIds = re.findall('=n([0-9]*)', html)

	if len(sysnetIds) == 0:
		return

	request_url = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n' + sysnetIds[0]
	request = urllib2.Request(request_url, None, headers=urlopenheader)
	response = urllib2.urlopen(request, timeout=5)
	html = response.read().decode('utf8')

	links = html.split()
	if len(links) == 0:
		return

	for link in links:
		yield link

def fetch_images_from_Google(keyword, use_safe_search=True, APIkey=None, cxID=None):
	from googleapiclient.discovery import build

	startId = 1
	errCnt = 0
	while (startId <= 91):
		try:
			service = build("customsearch", "v1",
							developerKey=APIkey)
			result = service.cse().list(
				q=keyword, cx=cxID, searchType='image', imgType='photo', start=int(startId)
			).execute()
		except Exception as e:
			print e
			errCnt += 1
			if errCnt > 10:
				print 'Error limit reached'
				return
			continue

		for i in result['items']:
			yield i['link']
		startId += len(result['items'])

if __name__ == "__main__":
	print 'Run downloader.py'
	exit(-1)
