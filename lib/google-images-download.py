import time 
import sys 
import os
import argparse
import ssl
import uuid
import ctypes
import multiprocessing
import selenium
import logging
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Image_Scraping(object):
    """docstring for Image_Scraping"""
    def __init__(self, 
                search_keyword = [],
                output = "",
                limit = 20,
                thread = 2,
                scroll = 200,
                proxy = "",
                color = "",
                image_type = ""):
        super(Image_Scraping, self).__init__()
        self.err = 0
        self.fin = 0
        self.search_keyword = search_keyword
        self.output = output
        self.max = limit
        self.thread = thread
        self.scroll = scroll
        self.proxy_ip, self.proxy_port = zip(proxy.split(":")[-2:])
        self.color = color
        self.image_type = image_type
        if self.proxy_ip:
            self.proxyDict = { 
                              "http"  : str(self.proxy_ip[0]), 
                              "https" : str(self.proxy_ip[0])
                              }

            #Request urlopen
            os.environ["http_proxy"]=proxy
            os.environ["https_proxy"]=proxy
        

    def get_folder(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != 17:
                return 
        else:
            return True

    def collector(self, url, html_link_queue, end):

        browser = webdriver.Firefox

        if self.proxy_ip:
            capabilities = webdriver.DesiredCapabilities().FIREFOX
            capabilities["marionette"] = False
            fp = webdriver.FirefoxProfile()
            # Direct = 0, Manual = 1, PAC = 2, AUTODETECT = 4, SYSTEM = 5
            fp.set_preference("network.proxy.type", 1)
            fp.set_preference("network.proxy.http", self.proxyDict["http"])
            fp.set_preference("network.proxy.http_port",int(self.proxy_port[0]))
            fp.set_preference("network.proxy.ssl", self.proxyDict["http"])
            fp.set_preference("network.proxy.ssl_port",int(self.proxy_port[0]))
            fp.set_preference("network.proxy.ftp", self.proxyDict["http"])
            fp.set_preference("network.proxy.ftp_port",int(self.proxy_port[0]))
            fp.update_preferences()
            browser = webdriver.Firefox(firefox_profile=fp, capabilities=capabilities)

        browser.get(url)

        while True:
            browser.execute_script("window.scrollBy(0,{0})".format(int(self.scroll)))
            for x in browser.find_elements_by_xpath("//a[@class='rg_l']"):
                xx = x.get_attribute("href")
                if xx:
                    html_link_queue.put(xx)
            if end.value:
                break
        browser.close()


    def download_worker(self, args):
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

                req = Request(img_abs_link,
                                headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                response = urlopen(req)
                #if args.endless:
                output_file = open(_file, 'wb')
                data = response.read()
                output_file.write(data)
                response.close()

                logger.info("thread {0} did {1} images".format(thread, self.fin))
                self.fin += 1

            except IOError as e:  # If there is any IOError
                logger.debug("failed IOError on image {0}".format(e))
                self.err += 1

            except HTTPError as e:  # If there is any HTTPError
                logger.debug("failed HTTPError  {0}".format(e))
                self.err += 1
            except URLError as e:
                logger.debug("failed URLError {0}".format(e))
                self.err += 1
            except ssl.CertificateError as e:
                logger.debug("failed CertificateError {0}".format(e))
                self.err += 1
            except Exception as e:
                logger.debug("failed {0}".format(e))
                self.err += 1

            if end.value:
                break

    def end_session(self, running_process, pool, *args):
        logger.info("tearDown")
        for p in running_process:
            logger.info("kill {0}".format(p))
            p.join()
        pool.close()
        logger.info("close pool {0}".format(pool))

    def scraping(self, *args):
        i = 0
        while i < len(self.search_keyword):
            search_term = self.search_keyword[i]
            search = search_term.replace(' ', '%20')
            if self.output:
                if not os.path.exists(self.output): raise Exception("path not valid!!!")
                dir_name = os.path.join(self.output, search_term)
            else:
                #calc default path
                file_path = os.path.dirname(os.path.realpath(__file__))
                dir_name = new_path = os.path.join(file_path[:-len(os.path.basename(file_path))],\
                 "data/raw_downloaded_image/{0}".format("{0}-{1}".format(search_term, str(self.color)) if self.color\
                  else search_term))

            if not os.path.exists(dir_name):
                if not self.get_folder(dir_name):
                    raise Exception("cant create folder!!!")
            else:
                if os.listdir(dir_name):
                    dir_name = os.path.join(dir_name, str(uuid.uuid4()))
                    if not self.get_folder(dir_name):
                        raise Exception("cant create folder!!!")

            _type = "tbs=itp:{0}".format(self.image_type)
            logger.info("Saving Files in {0}".format(dir_name))
            color_param = ('&tbs=ic:specific,isc:' + self.color) if self.color else ''
            url = 'https://www.google.com/search?q=' + search + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&'+ _type +'&tbm=isch' + color_param + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
            logger.info("Starting Download Process...")

            manager = multiprocessing.Manager()
            lifetime_end = manager.Value(ctypes.c_char_p, False)
            running_process = []
            html_link_queue = manager.Queue()    
            link_queue = manager.Queue()  
            search_links = multiprocessing.Process(target=self.collector, args=(url, html_link_queue,lifetime_end, ))
            running_process.append(search_links)
            search_links.start()

            pool = multiprocessing.Pool()
            threads = self.thread
            result = pool.map_async(self.download_worker, [(lifetime_end, link_queue, "Thread {0}".format(n), dir_name) for n in range(int(threads))])
            while len(os.listdir(dir_name))<=int(self.max):
                try:
                    link = html_link_queue.get()
                    if link_queue.empty():
                        link_queue.put(link)
                except KeyboardInterrupt as e:
                    lifetime_end.value = True
                    self.end_session(running_process,
                                     pool)
                    break
                     
                except Exception as e:
                    logger.debug(e)

            lifetime_end.value = True
            self.end_session(running_process,
                             pool)

            i = i + 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keywords', help='delimited list input', type=str, required=True)
    parser.add_argument('-o', '--output', help='output directory', type=str, required=False)
    parser.add_argument('-m', '--max', help='maximal download images', type=int, required=False, default=1000)
    parser.add_argument('-t', '--thread', help='download workers range', type=int, required=False, default=6)
    parser.add_argument('-s', '--scroll', help='scroll range', type=int, required=False, default=1000)
    parser.add_argument('-p', '--proxy', help='proxy ip:port', type=str, required=False)
    parser.add_argument('-y', '--type', help='search image type', type=str, required=False, choices=['face', 'clipart', 'photo', 'lineart', 'animated'])
    parser.add_argument('-c', '--color', help='filter on color', type=str, required=False, choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
    parser.add_argument('-v', '--verbose', action="store", help="0:Error, 1:Warning, 2:INFO*(default), 3:debug", default=2, type=int)
    args = parser.parse_args()
    search_keyword = [str(item) for item in args.keywords.split(',')]

    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[args.verbose])

    search = Image_Scraping(search_keyword,
                            args.output,
                            args.max,
                            args.thread,
                            args.scroll,
                            args.proxy,
                            args.color,
                            args.type)
    search.scraping()