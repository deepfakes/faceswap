import time 
import sys 
import os
import argparse
import ssl
import uuid
import ctypes
import multiprocessing
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Taking command line arguments from users
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keywords', help='delimited list input', type=str, required=True)
parser.add_argument('-o', '--output', help='output directory', type=str, required=False)
parser.add_argument('-m', '--max', help='maximal download images', type=int, required=False, default=1000)
parser.add_argument('-t', '--thread', help='download workers range', type=int, required=False, default=6)
parser.add_argument('-s', '--scroll', help='scroll range', type=int, required=False, default=1000)
parser.add_argument('-c', '--color', help='filter on color', type=str, required=False, choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
args = parser.parse_args()
search_keyword = [str(item) for item in args.keywords.split(',')]

# This list is used to further add suffix to your search term. Each element of the list will help you download 100 images. First element is blank which denotes that no suffix is added to the search keyword of the above list. You can edit the list by adding/deleting elements from it.So if the first element of the search_keyword is 'Australia' and the second element of keywords is 'high resolution', then it will search for 'Australia High Resolution'
keywords = [' high resolution']

def get_folder(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != 17:
            return 
    else:
        return True

version = (3,0)
cur_version = sys.version_info
if cur_version >= version:  # If the Current Version of Python is 3.0 or above
    # urllib library for Extracting web pages
    import urllib
    from urllib.request import Request, urlopen
    from urllib.request import URLError, HTTPError

else:  # If the Current Version of Python is 2.x
    # urllib library for Extracting web pages
    import urllib2
    from urllib2 import Request, urlopen
    from urllib2 import URLError, HTTPError

errorCount = 0
i = 0
def collector(url, html_link_queue, end):
    browser = webdriver.Firefox()
    browser.get(url)
    main_mem = []
    while True:     
        browser.execute_script("window.scrollBy(0,{0})".format(int(args.scroll)))
        mem = []
        for x in browser.find_elements_by_xpath("//a[@class='rg_l']"):
            xx = x.get_attribute("href")
            if xx:
                html_link_queue.put(xx)

        if end.value:
            break
    browser.close()


def download_worker(args):
    end = args[0]
    html_link_queue = args[1]
    thread = args[2]
    dir_name = args[3]
    while True:
        link = html_link_queue.get()
        try:
            img_abs_link = link.split("=")[1].replace("%2F", "/").replace("%3A", ":").split("&")[0]
            image_name = os.path.basename(img_abs_link).split(".")
            if len(image_name)==2: image_name, ext = zip(image_name)
            else: 
                ext = ["jpg"]
            _file = "{0}.{1}".format(str(image_name[0]),str(ext[0]))
            _file = os.path.join(dir_name, _file)
            if os.path.exists(_file):
                continue
            req = Request(img_abs_link, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
            response = urlopen(req, None, 15)
            #if args.endless:
            output_file = open(_file, 'wb')
            data = response.read()
            output_file.write(data)
            response.close()

            print("theard ", thread,"completed ====> ", str(_file))

        except IOError as e:  # If there is any IOError
            print("failed IOError on image ", e)

        except HTTPError as e:  # If there is any HTTPError
            print("failed HTTPError  ", e)

        except URLError as e:
            print("failed URLError ", e)

        except ssl.CertificateError as e:
            print("failed CertificateError ", e)       

        except Exception as e:
            print (e)

        if end.value:
            break


while i < len(search_keyword):
    search_term = search_keyword[i]
    search = search_term.replace(' ', '%20')
    if args.output:
        if not os.path.exists(args.output): raise Exception("path not valid!!!")
        dir_name = os.path.join(args.output, search_term)
    else:
        #calc default path
        file_path = os.path.dirname(os.path.realpath(__file__))
        dir_name = new_path = os.path.join(file_path[:-len(os.path.basename(file_path))],\
         "data/raw_downloaded_image/{0}".format("{0}-{1}".format(search_term, str(args.color)) if args.color\
          else search_term))

    if not os.path.exists(dir_name):
        if not get_folder(dir_name):
            raise Exception("cant create folder!!!")
    else:
        if os.listdir(dir_name):
            dir_name = os.path.join(dir_name, str(uuid.uuid4()))
            if not get_folder(dir_name):
                raise Exception("cant create folder!!!")

    print ("Saving Files in {0}".format(dir_name))
    color_param = ('&tbs=ic:specific,isc:' + args.color) if args.color else ''
    url = 'https://www.google.com/search?q=' + search + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + color_param + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
    print ("Starting Download Process...")

    manager = multiprocessing.Manager()
    #pool = multiprocessing.Pool()
    lifetime_end = manager.Value(ctypes.c_char_p, False)
    running_process = []
    html_link_queue = manager.Queue()    
    link_queue = manager.Queue()  
    search_links = multiprocessing.Process(target=collector, args=(url, html_link_queue,lifetime_end, ))
    running_process.append(search_links)
    search_links.start()

    download_workers = multiprocessing.Pool()
    threads = args.thread
    result = download_workers.map_async(download_worker, [(lifetime_end, link_queue, "Thread {0}".format(n), dir_name) for n in range(int(threads))])
    while len(os.listdir(dir_name))<=int(args.max):
        try:
            link = html_link_queue.get()
            if link_queue.empty():
                link_queue.put(link)
        except KeyboardInterrupt as e:
            print (e)
            lifetime_end.value = True
            print ("clean up")
            for p in running_process:
                print ("collector end ", p)
                p.join()
            download_workers.close()
            print ("download_workers end ")
        except Exception as e:
            print (e)

    lifetime_end.value = True
    for p in running_process:
        print ("collector end ", p)
        p.join()
    download_workers.close()
    print ("download_workers end ")

    i = i + 1

