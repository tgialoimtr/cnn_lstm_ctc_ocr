'''
Created on Feb 12, 2018

@author: loitg
'''
from __future__ import print_function
import cv2
import os, sys
import logging
from logging.handlers import TimedRotatingFileHandler
from common import args
from multiprocessing import Process, Manager, Pool
import json
from extract_fields.extract import CLExtractor
from column import Store, Column
import numpy as np
import pandas as pd
import codecs


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
    img = cv2.imread(imgpath)
    h, w = img.shape[:2]
    newwidth = int(950.0 * w / h)
    if newwidth > 300:
        img = cv2.resize(img, (newwidth, 950))
    else:
        img = cv2.resize(img, (300, int(300.0 * h / w)))
    cv2.imshow('kk', img)
    cv2.waitKey(500)

s = '=-=++-='

def createColumns(infopath):
    infos = json.load(open(infopath,'r'))
    data = []
    for info in infos:
        data.append(Store(info))
    mallCol = Column(Store.MALL_NAME, 2.0, 8.0)
    storeCol = Column(Store.STORE_NAME, 2.0, 8.0)
    for store in data:
        mallCol.initAddRow(store)
        storeCol.initAddRow(store)
    print(str(len(data)) + '----' + str(len(storeCol.values.keys())))  
    return storeCol, mallCol

def createColumnsCsv(infopath):
    temp = pd.read_csv(infopath, dtype={'zipcode':np.str})
    temp.fillna('', inplace=True)
    temp = temp.to_dict('records')
    data = []
    for s in temp:
        data.append(Store(s))
    mallCol = Column(Store.MALL_NAME, 2.0, 8.0)
    storeCol = Column(Store.STORE_NAME, 2.0, 8.0)
    for store in data:
        mallCol.initAddRow(store)
        storeCol.initAddRow(store)
    print(str(len(data)) + '----' + str(len(storeCol.values.keys())))  
    return storeCol, mallCol
     
def suggestLC(storecol, lines):
    {u'storeName': u'AE BY SPORTSLINK AND 12TH MAN', u'mallName': u'BEDOK', u'zipcode': 467360, u'gstNoPattern': u'199400069M', u'receiptIdPattern': u'INVOICE:|NULL|NULL', u'receiptIdLastToken': u':', u'locationCode': u'L24066090381'}
    rs_store = storecol.search0(lines)
#     for store, val in rs_store:
#         print('' %()) str(val) + '--' + store.storedict['mallName'] + '--' + store.storedict['storeName'])
    rs_store = sorted(rs_store, key=lambda x:(x[1], x[0].storeKeyword, x[0].mallKeyword))
    return [x[0] for x in rs_store]

def locodeExist(topx00path, locodes):
    lcs = [line.rstrip().split(',')[1] for line in open(topx00path, 'r')]
    inlocodes = [x in lcs for x in locodes]
    return inlocodes


def appendToTop(topx00path, newlc):
    with open(topx00path, 'a') as of:
        of.write(newlc['rank'] + ',' + newlc['code'] + ',,'  + newlc['mall'] + ','  + newlc['store'] + ','  + newlc['zipcode'] + ','  + newlc['gst'] + '\n' )


logger = createLogger('main')
largedata = '/root/build_data/13kreceipts/'
textspath = '/root/build_data/texts/'
infopath = '/root/build_data/trung_kw_3.csv'

try:
    configlines =  list(open('./.abc.txt', 'r'))
except IOError:
    configlines = []

topx00path = os.path.join(args.javapath, args.dbfile)
    
def nextImage(fn, show=False):
    lines = list(open(os.path.join(textspath, fn), 'r'))
    locode0 = extractor.locode_extr.extract(lines[:])
    locs = locode0.split('=,=')
    if len(locs) == 5:# SKIP
        print('%-15s:%s' % ('Location Code', locs[0]))
        print('%-15s:%s' % ('Mall', locs[1]))
        print('%-15s:%s' % ('Store', locs[2]))
        print('%-15s:%s' % ('GST No', locs[3]))
        print('%-15s:%s' % ('ZipCode', locs[4]))
        return True
    else: #unable to find
        #display image
#         fn = fn[:-4]  
#         show(os.path.join(largedata, fn))
        return False

def enter(name, value=None):
    if name == 'code':
        fullName = 'LocationCode'
    elif name == 'name':
        fullName = 'LocationName'
    elif name == 'rank':
        fullName = 'Rank'
    elif name == 'gst':
        fullName = 'GST No'
    elif name == 'zipcode':
        fullName = 'ZipCode'
    elif name == 'store':
        fullName = 'Store Name'
    elif name == 'mall':
        fullName = 'Mall Name'
    print("%-11s: " % fullName, end='')
    if value is None:
        user_input = sys.stdin.readline().rstrip()
        return user_input
    else:
        print(value)
        return value
    


if __name__ == '__main__':
    groupbypath = sys.argv[1]
    top600path = sys.argv[2]
    print(top600path)
    print(groupbypath)
    
    top600 = {}
    abc = []
    for i, line in enumerate(codecs.open(top600path, 'r', 'utf8')):
        temp = line.split(',')
        top600[temp[0]] = {}
        top600[temp[0]]['name'] = temp[1]
        top600[temp[0]]['rank'] = str(i)
        abc.append((i, temp[0]))
    
    topx00 = {}
    for line in open(topx00path, 'r'):
        temp = line.split(',')
        temp[1] = temp[1].split('_')[0]
        topx00[temp[1]] = {}
        topx00[temp[1]]['rank'] = temp[0]
        topx00[temp[1]]['code'] = temp[1]
        topx00[temp[1]]['mall'] = temp[2]
        topx00[temp[1]]['store'] = temp[3]
        topx00[temp[1]]['gst'] = temp[5]
        topx00[temp[1]]['zipcode'] = temp[4]
        
    
    for _, locode in sorted(abc):
        filelist = []
        current_index = 0
        locodepath = os.path.join(groupbypath, locode)
        if not os.path.isdir(locodepath): continue
        for fn in os.listdir(locodepath):
            if fn[-3:].upper() in ['PEG', 'JPG']:
                filelist.append(fn)      
        if locode in topx00:
            continue
#             enter('code', locode)
#             enter('name', top600[locode]['name'])
#             enter('rank', top600[locode]['rank'])
#             enter('mall', topx00[locode]['mall'])
#             enter('store', topx00[locode]['store'])
#             enter('zipcode', topx00[locode]['zipcode'])
#             enter('gst', topx00[locode]['gst'])
#             show(os.path.join(locodepath, filelist[current_index]))
#             while True:
#                 k = raw_input('1,3, (N)ext: ')
#                 if k == '1':
#                     current_index = current_index - 1 if current_index > 0 else len(filelist) - 1
#                     show(os.path.join(locodepath, filelist[current_index]))
#                 elif k == '3':
#                     current_index = current_index + 1 if current_index < len(filelist) - 1 else 0
#                     show(os.path.join(locodepath, filelist[current_index]))
#                 elif k.upper() == 'N':
#                     break
        else:
            kw = {}
            kw['code'] = enter('code', locode)
            kw['name'] = enter('name', top600[locode]['name'])
            kw['rank'] = enter('rank', top600[locode]['rank'])
            show(os.path.join(locodepath, filelist[current_index]))
            while True:
                k = raw_input('1,3, (I)nput, (N)ext: ')
                if k == '1':
                    current_index = current_index - 1 if current_index > 0 else len(filelist) - 1
                    show(os.path.join(locodepath, filelist[current_index]))
                elif k == '3':
                    current_index = current_index + 1 if current_index < len(filelist) - 1 else 0
                    show(os.path.join(locodepath, filelist[current_index]))
                elif k.upper() == 'I':
                    kw['mall'] = enter('mall')
                    kw['store'] = enter('store')
                    kw['zipcode'] = enter('zipcode')
                    kw['gst'] = enter('gst')
                    while True:
                        print('Edit code, mall, store, zipcode, gst, or done ? ', end='')
                        editwhat = sys.stdin.readline().rstrip()
                        if editwhat.lower().strip() in ['d', 'done']:
                            appendToTop(topx00path, kw)
                            break
                        elif editwhat.lower().strip() in ['code', 'gst', 'zipcode', 'mall', 'store']:
                            kw[editwhat.lower().strip()] = enter(editwhat.lower().strip())
                    break    
                elif k.upper() == 'N':
                    break 
