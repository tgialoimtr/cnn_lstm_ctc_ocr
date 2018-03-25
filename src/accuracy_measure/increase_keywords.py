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
from column import Store, Column


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
        
def suggestLC(storecol, mallcol, lines):
    {u'storeName': u'AE BY SPORTSLINK AND 12TH MAN', u'mallName': u'BEDOK', u'zipcode': 467360, u'gstNoPattern': u'199400069M', u'receiptIdPattern': u'INVOICE:|NULL|NULL', u'receiptIdLastToken': u':', u'locationCode': u'L24066090381'}
    rs_store = storecol.search0(lines)
    rs_mall = mallcol.search0(lines)
    print('store: ' + str(len(rs_store)))
    print('mall: ' + str(len(rs_mall)))
    return []

def updateTopX00():
    pass

if __name__ == '__main__':
    logger = createLogger('main')
    largedata = '/tmp/textresult/13kreceipts/'
    textspath = '/tmp/textresult/texts/'
    infopath = '/tmp/textresult/database.json'

    storecol, mallcol = createColumns(infopath)
    
    
    topx00path = os.path.join(args.javapath, args.dbfile)
    currentfile = None #'1501685782273_b5e7312d-fabc-4203-bf1e-6ae4f468d6f1.JPG'
    extractor = CLExtractor()
        
    # loop through folder to find locationcode from currentfile
    for fn in os.listdir(textspath):
        if currentfile is not None:
            if fn != currentfile:
                continue
            else:
                currentfile = None
        
        lines = list(open(os.path.join(textspath, fn), 'r'))
        locode0 = extractor.locode_extr.extract(lines[:])
        locs = locode0.split('=,=')
        if len(locs) == 5:
            print(str(locs))
        else: #unable to find
            suggestions = suggestLC(storecol, mallcol, lines[:])
            #display suggestion
            for i, lc in enumerate(suggestions):
                print('%d: %s', i, lc.toLine())
                
                
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
        
            
        
        
        
        
        
                     
                




