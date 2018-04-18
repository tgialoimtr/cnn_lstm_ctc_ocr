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
        of.write(',' + newlc['code'] + ',,'  + newlc['mall'] + ','  + newlc['store'] + ','  + newlc['zipcode'] + ','  + newlc['gst'] + '\n' )


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

def enter(name):
    if name == 'code':
        fullName = 'LocationCode'
    elif name == 'gst':
        fullName = 'GST No'
    elif name == 'zipcode':
        fullName = 'ZipCode'
    elif name == 'store':
        fullName = 'Store Name'
    elif name == 'mall':
        fullName = 'Mall Name'
    print("%-11s: " % fullName, end='')
    user_input = sys.stdin.readline().rstrip()
    return user_input


if __name__ == '__main__':
    storecol, _ = createColumnsCsv(infopath)
    currentfile = configlines[0].rstrip() if len(configlines) > 0 else None
    extractor = CLExtractor()
    textfiles = sorted(os.listdir(textspath))
    currentfile_index = textfiles.index(currentfile) if currentfile else 0
    
    try:
        skip_known = False
        to_right = True
        #new loop
        while True:
            locodefound = nextImage(textfiles[currentfile_index])
            fn = textfiles[currentfile_index][:-4]
            print('Current file: ' + fn + '-----------------')
            if skip_known and locodefound: 
                if to_right:
                    if currentfile_index < len(textfiles) - 1: currentfile_index += 1
                else:
                    if currentfile_index > 0: currentfile_index -= 1
                continue
            fn = os.path.join(largedata, fn) 
            show(fn)
            while True:
                print('Current file: ' + fn + '-----------------')
                print('A,D,1,3, (T)ext, (F)ind, (I)nput: ', end = '')
                k = sys.stdin.readline().rstrip()
                if k == 'a':
                    skip_known = True
                    to_right = False
                    if currentfile_index > 0: currentfile_index -= 1
                    break
                elif k == 'd':
                    skip_known = True
                    to_right = True
                    if currentfile_index < len(textfiles) - 1: currentfile_index += 1
                    break
                elif k == '1':
                    skip_known = False
                    to_right = False
                    if currentfile_index > 0: currentfile_index -= 1
                    break
                elif k == '3':
                    skip_known = False
                    to_right = True
                    if currentfile_index < len(textfiles) - 1: currentfile_index += 1
                    break
                elif k == 't':
                    with open(os.path.join(textspath, textfiles[currentfile_index]), 'r') as f:
                        for line in f:
                            print(line.rstrip())
                elif k == 'f':
                    print('store name to find locode: ', end = '')
                    kws = sys.stdin.readline().rstrip()
                    if kws == 'n': break
                    kws = kws.split(',')
                    suggestions = suggestLC(storecol, kws)
                    locodes = [sgt.locationCode for sgt in suggestions]
                    intopx00 = locodeExist(topx00path, locodes)
                    
                    for i, (lc, intop) in enumerate(zip(suggestions, intopx00)):
                        if intop:
                            print('IN-DB', end='')
                        else:
                            print('     ', end = '')
                        print('%2d: %s' % ( i, lc.toString()))
                elif k == 'i':
                    kw = {}
                    kw['code'] = enter('code')
                    kw['mall'] = enter('mall')
                    kw['store'] = enter('store')
                    kw['zipcode'] = enter('zipcode')
                    kw['gst'] = enter('gst')
                    while True:
                        print('Edit code, mall, store, zipcode, gst, or done ? ', end='')
                        editwhat = sys.stdin.readline().rstrip()
                        if editwhat.lower() in ['d', 'done']:
                            appendToTop(topx00path, kw)
                            break
                        elif editwhat.lower() in ['gst', 'zipcode', 'mall', 'store']:
                            kw[editwhat] = enter(editwhat)                            
                else:
                    print('Unknown Command.')
    
    except KeyboardInterrupt:
        print()
        with open('./.abc.txt', 'w') as of:
            of.write(textfiles[currentfile_index] + '\n')
        sys.exit(0)
    
    
    
    
    
    
     
     
     
     
     
     
     
    # loop through folder to find locationcode from currentfile
#     for fn in os.listdir(textspath):
#         if currentfile is not None:
#             if fn != currentfile:
#                 continue
#             else:
#                 currentfile = None
#          
#         lines = list(open(os.path.join(textspath, fn), 'r'))
#         locode0 = extractor.locode_extr.extract(lines[:])
#         locs = locode0.split('=,=')
#         if len(locs) == 5:# SKIP
#             print(str(locs))
#         else: #unable to find
#             #display image
#             fn = fn[:-4]  
#             show(os.path.join(largedata, fn))
#             lcid = 'n'
#             while True:
#                 kws = raw_input('store name to find locode: ')
#                 if kws == 'n': break
#                 kws = kws.split(',')
#                 suggestions = suggestLC(storecol, kws)
#                 if len(suggestions) == 0: continue
#                 for i, lc in enumerate(suggestions):
#                     print('%d: %s' % ( i, lc.toString()))
#                 lcid = raw_input('locationcode id: ')
#                 if lcid == 'n': break
#                 try:
#                     lcid = int(lcid)
#                     if lcid >= 0: break
#                 except Exception:
#                     pass
#             if lcid == 'n': continue
#             newlc = suggestions[lcid]
#             # find locationcode exist in topx00 or not
#             if not locodeExist(topx00path, newlc.locationCode):
#                 #input new gst, zipcode, store mall for lcid
#                 newlc.storeKeyword = raw_input('store keyowrds: ')
#                 if newlc.storeKeyword == 'n': continue
#                 newlc.mallKeyword = raw_input('mall keyowrds: ')
#                 if newlc.mallKeyword == 'n': continue
#                 newlc.gst = raw_input('gst: ')
#                 if newlc.gst == 'n': continue
#                 newlc.zipcode = raw_input('zipcode: ')
#                 if newlc.zipcode == 'n': continue
#                 appendToTop(topx00path, newlc)
            
#a d:normal
#A D:skip known  # also SKIP unkown usually repeated.
##: integrate 5000 _ 200; handel 10%
##: input ground truth (only location ?) while build kw, also mechanism for fast automactically test.
##: math CRM(later),        
    
           
# if __name__ == '__main__':
#     allrs = {}
#     currentfile = '1501685782273_b5e7312d-fabc-4203-bf1e-6ae4f468d6f1.JPG'
#     with open('/tmp/temp/rs.txt','r') as f:
#         fn = None
#         for line in f:
#             temp = line.split('----------------')
#             if '.JPG----------------' in line and len(temp) > 1:
#                 fn = temp[0]
#                 allrs[fn] = []
#             else:
#                 rs = re.findall(r'<== (.*?) == (.*?) == (.*?) == >', line)
#                 allrs[fn].append(rs)
#                 
#     with open('/tmp/temp/measure.csv','a') as of:
#         for fn, predlines in allrs.iteritems():
#             if currentfile is not None:
#                 if fn != currentfile:
#                     continue
#                 else:
#                     currentfile = None
#             i = 0
#             preds = {}
#             print(Fore.RED + fn)
#             for line in predlines:
#                 for pred in line:
#                     preds[i] = pred
#                     print(Fore.RED + str(i) +' ', end = '')
#                     i += 1
#                     print(Fore.BLACK + pred[0]+ ' ', end = '')
#                 print('')
#              
#             show('/home/loitg/part1/'+fn)
#             bad = raw_input('bad image ? (y/n) ')
#             buff = ''
#             while True:
#                 j = raw_input('which line ? (n to next, u to undo)') 
#                 if j != 'u' and buff != '':
#                     of.write(buff)
#                 if j == 'n': break
#                 if j == 'u': j = raw_input('which line ? (n to next, u to undo)') 
#                 j = int(j)
#                 linetype = raw_input('k(eyword), p(rice), d(ate), b(oth): ')
#                 gt = raw_input('ground truth (blank if correct):')
#                 if gt == '':
#                     gt = preds[j][0]
#                     sol = ''
#                 else:
#                     sol = raw_input('special reason ? (badimg, preprocess, trainfont, regex, leave blank if correct) ')
#                 if j >= 0:
#                     buff = fn + s + bad +s+ linetype + s +sol + s + gt + s + preds[j][0] + s + preds[j][1] + s + preds[j][2] + '\n'
#                     # fn = temp[0]; bad = temp[1]; linetype=temp[2]; sol=temp[3]; gt=temp[4]; preds=(temp[5], temp[6],temp[7]);
#                 else:
#                     buff = fn + s + bad +s+ linetype + s +sol + s + gt + s + s + s + '\n'
#             of.flush()
 
 

#     manager = Manager()
#     states = manager.dict()
#     server = LocalServer(args.model_path, manager)
#     serverprocess = Process(target=runserver, args=(server, states)) 
#     processes = []
#     process_args = []
# 
#     batches = []
#     for fn in sorted(os.listdir(largedata)):
#         if currentfile is not None:
#             if fn != currentfile:
#                 continue
#             else:
#                 currentfile = None
#              
#         allimgs.append(fn)
#     allimgs.sort()
    
    
    # there may be a QUEUE, push results of batch into here   
    
#     for k in os.listdir('/home/loitg/part2/'):  
#         print k 
#         while True:
#             show('/home/loitg/part2/'+k,2)
#             val = raw_input('total: ')
#             command= raw_input('what next ? ')
#             if command != 'reshow':
#                 print k + ':' + val
#                 break
        
            
        
        
        
        
        
                     
                




