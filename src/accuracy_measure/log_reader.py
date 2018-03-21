'''
Created on Mar 16, 2018

@author: loitg
'''

import sys
import re
import json
import csv

def queuePush2Csv(logfilepath, csvpath):
#     rpushmsglog = r'\d+?-\d+?-\d+? (\d+?:\d+?:\d+?(,\d+?)?) \[[\w\d ]+?\] \[.+?\] \d+?, (\{.*\})'
    rpushmsglog = r'\d+?-\d+?-\d+?[ ]*(\d+?:\d+?:\d+?(,\d+?)?)[ ]*?\[.*?\][ ]*?\[.*?\][ ]*?\d+?,[ ]*?(\{.*\})'
    pushmsglog = re.compile(rpushmsglog)
    logfile = open(logfilepath, 'r')     
    keys = ["status", "deviceName", "zipcode", "storeName", "receiptBlobName", "station", "mallName", "amount", "mobileVersion", "currency", "token", 
            "program", "gstNo", "totalNumber", "receiptCrmName", "memberNumber", "receiptDateTime", "receiptId", "locationCode", "uploadLocalFolder"]   

    with open(csvpath, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        for line in logfile:
            r = pushmsglog.match(line)
            if r:
                print r.group(3)
                newrow = json.loads(r.group(3))
                for k in newrow:
                    if type(newrow[k]) is unicode:
                        newrow[k] = newrow[k].encode('ascii','ignore')
                print type(newrow['currency'])
                print newrow
                dict_writer.writerow(newrow)
                

if __name__ == '__main__':
#     sys.argv = ['main.py','/home/loitg/location_nn/logs/log.worker-0', '/tmp/push_result.csv']
    print sys.argv[1]
    print sys.argv[2]
    queuePush2Csv(sys.argv[1], sys.argv[2])
    
    
