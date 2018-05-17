#!/usr/bin/env python
import sys
import os
from time import sleep
from multiprocessing import Process, Manager, Pool
import logging
from logging.handlers import TimedRotatingFileHandler
from azure.common import AzureException, AzureMissingResourceHttpError
import argparse
import csv
import simplejson as json

from processing.pagepredictor import PagePredictor
from processing.server import LocalServer
from extract_fields.extract import CLExtractor
from inputoutput.receipt import ExtractedData, ReceiptSerialize
from inputoutput.azureservice import AzureService
from common import args
from base64 import b64encode
from time import time

parser = argparse.ArgumentParser("OCR-App for receipts")
# local processing
parser.add_argument("-i","--input",default=args.imgsdir,
                    help="input images, can be a directory or a single file")
parser.add_argument("-o","--output",default="./result.csv",
                    help="output csv path")
cmd_args = parser.parse_args()


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

def ocrLocalPath(reader, num, states):
    logger = createLogger('local-' + str(num))
    logger.info('process %d start pushing image.', num)
    if os.path.isdir(cmd_args.input):
        imglist = sorted(os.listdir(cmd_args.input), reverse=False)
    else:
        imglist = [cmd_args.input]
    keys = ["status", "deviceName", "zipcode", "storeName", "receiptBlobName", "station", "mallName", "amount", "mobileVersion", "currency", "token", 
        "program", "gstNo", "totalNumber", "receiptCrmName", "memberNumber", "receiptDateTime", "receiptId", "locationCode", "uploadLocalFolder", "qualityCode"]
    extractor = CLExtractor()
    with open(cmd_args.output, 'w') as outfile:
        dict_writer = csv.DictWriter(outfile, keys)
        dict_writer.writeheader()      
    for filename in imglist:
        states[num] += 1
        if filename[-3:].upper() in ['JPG', 'PEG'] and hash(filename) % args.numprocess == num:
            try:
                lines, qualityCode = reader.ocrImage(os.path.join(args.imgsdir, filename), logger)
                extdata = extractor.extract(lines)
                extdata.qualityCode = qualityCode
            except Exception:
                logger.exception('EXCEPTION WHILE READING LINES.')
                extdata = ExtractedData()
                extdata.qualityCode = 0
            
            rinfo = ReceiptSerialize()
            rinfo.receiptBlobName = unicode(filename, 'utf-8')
            newrow = json.loads(rinfo.combineExtractedData(extdata))
            for k in newrow:
                if type(newrow[k]) is unicode:
                    newrow[k] = newrow[k].encode('ascii','ignore')
            with open(cmd_args.output, 'a') as outfile2:
                print ('write ' + str(newrow) + ' to ' + cmd_args.output)
                dict_writer = csv.DictWriter(outfile2, keys)
                dict_writer.writerow(newrow)
            
            if lines is None or len(lines) == 0: continue
            with open(args.textsdir + filename + '.txt', 'w') as outfile3:
                for line in lines:
                    outfile3.write(line + '\n')


if __name__ == "__main__":
    logger = createLogger('main')
    extractor = CLExtractor()
    with open('/tmp/result.txt', 'a') as rawmsgof:
        i = 0
        for fn in os.listdir(args.textsdir):
            if fn[-3:].lower() != 'txt': continue
            filename=fn[:-4]
            lines = []
            
            for line in open(os.path.join(args.textsdir, fn)):
                lines.append(line.strip())
            try:
                i += 1
                tt = time()
                print str(i)
                extdata = extractor.extract(lines)
                print str(i) + ':' + str(time() - tt)
            except Exception:
                logger.exception('EXCEPTION WHILE READING LINES.')
                extdata = ExtractedData()
                extdata.qualityCode = 0
            rinfo = ReceiptSerialize()
            rinfo.receiptBlobName = unicode(filename, 'utf-8')
            msg = rinfo.combineExtractedData(extdata)        
            rawmsgof.write(msg + '\n')
