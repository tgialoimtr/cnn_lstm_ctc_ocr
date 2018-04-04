'''
Created on Feb 12, 2018

@author: loitg
'''
from __future__ import print_function
import cv2
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from common import args
from multiprocessing import Process, Manager, Pool
import json
from extract_fields.extract import CLExtractor



def createLogger(name):
    logFormatter = logging.Formatter("%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(name)
    
    fileHandler = TimedRotatingFileHandler(os.path.join(args.logsdir, 'log.' + name) , when='midnight', backupCount=10)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger

def show(imgpath):
    print('showing' + imgpath)
    img = cv2.imread(imgpath)
    h, w = img.shape[:2]
    newwidth = int(950.0 * w / h)
    if newwidth > 300:
        img = cv2.resize(img, (newwidth, 900))
        cv2.imshow('kk', img)
        cv2.waitKey(500)
    else:
        os.system('xdg-open ' + imgpath)

s = '=-=++-='
 
class StatLineBase(object):
    def __init__(self, csvoutfile, *argv):
        self.csvoutfile = csvoutfile
        self.data = {}
        self.args = sorted(list(argv))
        for arg in self.args:
            print(arg + ', ', end='')
            self.data[arg] = None
        print()
        
    def input(self, appendtofile=True):
        for arg in self.args:
            self.data[arg] = raw_input(arg + ': ')
        if appendtofile:
            with open(self.csvoutfile, 'a') as of:
                for arg in self.args:
                    of.write(self.data[arg] + ',')
                of.write('\n')
        return self


if __name__ == '__main__':
    logger = createLogger('main')
    largedata = '/home/loitg/Downloads/22kreceipts/'
    textspath = '/home/loitg/Downloads/texts/'
    of = StatLineBase('/tmp/temp.txt', 'location', 'price', 'date')
    currentfile = None #'1501685782273_b5e7312d-fabc-4203-bf1e-6ae4f468d6f1.JPG'
        
    # loop through folder to find locationcode from currentfile
    for fn in os.listdir(textspath):
        if currentfile is not None:
            if fn != currentfile:
                continue
            else:
                currentfile = None
        
        lines = open(os.path.join(textspath, fn), 'r')
        for line in lines:
            print(line.rstrip())
        
        show(os.path.join(largedata, fn[:-4]))
        of.input()
        
        

























       
#         locode0 = extractor.locode_extr.extract(lines)
#         locs = locode0.split('=,=')
#         if len(locs) == 5:
#             print(str(locs))
#         else: #unable to find
#             suggestions = suggestLC(topx00path, lines)
#             #display suggestion
#             for i, lc in enumerate(suggestions):
#                 print('%d: %s', i, lc.toLine())
#             #display image
#             fn = fn[:-4]  
#             show(os.path.join(largedata, fn))
#             #
#             lcid = raw_input('locationcode id: ')
#             if lcid == 'n': continue
#             try:
#                 lcide = int(lcid)
#                 
#             except Exception:
#                 while True:
#                     kws = raw_input('find locationcode by: ')
#                     if kws == 'n': break
#                     kws = kws.split(',')
#                     suggestions = suggestLC(topx00path, lines)
#                     for i, lc in enumerate(suggestions):
#                         print('%d: %s', i, lc.toLine())
#                     lcid = raw_input('locationcode id: ')
#                     if lcid == 'n': break
#                     try:
#                         lcide = int(lcid) 
#                     except Exception:
#                         pass
#             if lcid == 'n': continue
#             newlc = suggestions[i]
#             #input new gst, zipcode, store mall for lcid
#             pass
#             appendToTop(topx00path, newlc)

