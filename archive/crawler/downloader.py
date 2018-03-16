#!/usr/bin/env python2
#-*- coding: utf-8 -*-
import os, sys, urllib2, threading, posixpath, urlparse, argparse, atexit, random, socket, time, hashlib, pickle, signal
import string
import imghdr
import imagesets

reload(sys)
sys.setdefaultencoding('utf-8')

#config
socket.setdefaulttimeout(2)
N_MAX_THREADS = 20

# For synchronization
working_thr_cnt = 0
t_cnt_lock = threading.Lock()
img_hash_lock = threading.Lock()

in_progress = []
downloaded_urls = []
failed_urls = []
image_md5s = {}
urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:42.0) Gecko/20100101 Firefox/42.0'}


def download(url, output_dir):
	global working_thr_cnt
	global downloaded_urls, failed_urls, image_md5s      # Is this necessary?

	if url in downloaded_urls:
		t_cnt_lock.acquire()
		working_thr_cnt += (-1)
		t_cnt_lock.release()
		return

	path = urlparse.urlsplit(url).path
	filename = posixpath.basename(path)

	# Filename manipulation
	filename = filename.split('?')[0]		# Strip GET parameters
	if len(filename)>40:
		name, ext = os.path.splitext(filename)
		filename = name[:36] + ext

	try:
		request=urllib2.Request(url,None,urlopenheader)
		image=urllib2.urlopen(request, timeout=5).read()
		if len(image)==0:
			print('no image')

		if not imghdr.what(None, image):
			print('FAIL Invalid image format')
			return

		# Add an extension if needed
		if os.path.splitext(filename)[1] == '':
			filename = filename + '.' + imghdr.what(None, image)

		while os.path.exists(os.path.join(output_dir, filename)):
			# FIXME: What if two or more threads generate the same random_str at the same time?
			random_str = ''.join(random.choice(string.ascii_letters + string.digits) for x in range(6))
			filename = random_str + filename

		in_progress.append(filename)

		md5 = hashlib.md5()
		md5.update(image)
		md5_key = md5.hexdigest()
		img_hash_lock.acquire()
		if md5_key in image_md5s:
			img_hash_lock.release()
			print('FAIL Image is a duplicate of ' + image_md5s[md5_key] + ', not saving ' + filename)
			in_progress.remove(filename)
			return

		image_md5s[md5_key] = filename
		img_hash_lock.release()

		imagefile=open(os.path.join(output_dir, filename),'wb')
		imagefile.write(image)
		imagefile.close()
		in_progress.remove(filename)

		print("OK " + filename)

		downloaded_urls.append(url)
	except Exception as e:
		print("FAIL " + filename)
		failed_urls.append((url, output_dir))
	finally:
		t_cnt_lock.acquire()
		working_thr_cnt += (-1)
		t_cnt_lock.release()

def removeNotFinished():
	for filename in in_progress:
		try:
			if os.path.exists(filename):
				os.remove(os.path.join(output_dir, filename))
		except IOError:
			pass

def download_images(imgURLs, output_dir):
	global working_thr_cnt

	# Download images
	for link in imgURLs:
		while (working_thr_cnt >= N_MAX_THREADS):
			time.sleep(0.1)

		# increment counter first
		t_cnt_lock.acquire()
		working_thr_cnt += 1
		t_cnt_lock.release()
		t = threading.Thread(target=download, args=(link, output_dir))
		t.start()

	while (working_thr_cnt > 0):
		time.sleep(0.1)

def backup_history(*args):
	download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'wb')
	pickle.dump(downloaded_urls,download_history)
	copied_image_md5s = dict(image_md5s)  # To resolve a concurrency issue
	pickle.dump(copied_image_md5s, download_history)
	download_history.close()
	print('history_dumped')
	if args:
		exit(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Bulk image downloader')
	parser.add_argument('-s', '--search-string', required = False,
						help = 'Keyword to search')
	parser.add_argument('-f', '--search-file', required = False,
						help = 'Path to a file containing search strings line by line')
	parser.add_argument('-o', '--output', required = False, default='./images',
						help='Output directory (./images by default)')
	parser.add_argument('-e', '--search-engine', choices=['bing', 'google', 'imagenet'], default='bing',
						help = 'Choose search engine to use (Bing by default)')
	parser.add_argument('--google_apikey', type=str, default=None, help = 'Google API key')
	parser.add_argument('--google_cx', type=str, default=None, help = 'Google search engine ID')
	parser.add_argument('--no-filter', action = 'store_true', required = False, default=False,
						help='Disable adult filter (if applicable)')
	args = parser.parse_args()
	if (not args.search_string) and (not args.search_file):
		parser.error('Provide either search string or path to file containing search strings')

	output_dir = args.output
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	base_output_dir = output_dir

	use_safe_search = not args.no_filter

	if args.search_file:
		try:
			inputFile = open(args.search_file)
			keywords = inputFile.read().splitlines()
			inputFile.close()
		except (OSError, IOError):
			print("Couldn't open file {}".format(args.search_file))
			exit(1)
		extend_dir = True
	else:
		keywords = [args.search_string]
		extend_dir = False

	atexit.register(removeNotFinished)
	signal.signal(signal.SIGINT, backup_history)

	# TODO: Move to a new function
	try:
		download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'rb')
		downloaded_urls=pickle.load(download_history)
		image_md5s=pickle.load(download_history)
		download_history.close()
	except (OSError, IOError):
		downloaded_urls=[]

	for keyword in keywords:
		if extend_dir:
			output_dir = os.path.join(base_output_dir, keyword.strip().replace(' ', '_'))
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# TODO: Use a fancier multi-thread model (e.g. Producer-Consumer)
		if args.search_engine == 'google':
			urlGen = imagesets.fetch_images_from_Google(keyword, use_safe_search,
														args.google_apikey, args.google_cx)
		elif args.search_engine == 'imagenet':
			urlGen = imagesets.fetch_images_from_ImageNet(keyword)
		else:
			urlGen = imagesets.fetch_images_from_Bing(keyword, use_safe_search)

		imgURLs= list(urlGen)

		download_images(imgURLs, output_dir)

		retryURLs = map(lambda x: x[0], failed_urls)
		download_images(retryURLs, output_dir)

		failed_urls=[]

		backup_history()