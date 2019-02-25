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

from processing.pagepredictor2 import PagePredictor
from processing.server import LocalServer
from extract_fields.extract import CLExtractor
from inputoutput.receipt import ExtractedData, ReceiptSerialize
from inputoutput.azureservice import AzureService
from common import args
from base64 import b64encode


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

def createAzureService(logger):
    try:
        service = AzureService(connection_string=args.connection_string,
                               container_name=args.container_name,
                               queue_get=args.queue_get_name,
                               queue_push=args.queue_push_name,
                               )
        return service
    except AzureException as e:
        logger.error('Connection Error: Maybe wrong credential.')
        return None
    
def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger)

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
            

    
def ocrQueue(reader, num, states):
    logger = createLogger('worker-' + str(num))
    logger.info('process %d start pushing image.', num)
    extractor = CLExtractor()
    service = createAzureService(logger)
            
    while True:  
        m, rinfo = service.getReceiptInfo(logger=logger)
        if m is not None: # got message
            if rinfo is not None: # parse success
                logger.info('message before parsed: %s', m.content)
                logger.info('message after parsed: %s', rinfo.toString())
                lp = service.getImage(rinfo.receiptBlobName, logger=logger)
                if lp is not None:
                    if len(lp) > 0:
                        extdata = None
                        try:
                            lines, qualityCode = reader.ocrImage(lp, logger)
                            logger.info('process %d has %d lines.',num, len(lines))
                            for line in lines:
                                logger.debug(line)
                            extdata = extractor.extract(lines)
                            extdata.qualityCode = qualityCode
                            states[num] += 1
                        except Exception:
                            logger.exception('EXCEPTION WHILE EXTRACTING LINES AND FIELDS.')
                            extdata = ExtractedData()
                            extdata.qualityCode = 0
                        try:
                            outmsg = rinfo.combineExtractedData(extdata)
                            logger.info('%d, %s', states[num], outmsg)
                            service.pushMessage(b64encode(outmsg).decode('utf-8'), logger=logger) # Fix bug b64-encode type of Azure
                            service.deleteMessage(m, logger=logger)
                            #service.deleteImage(rinfo.receiptBlobName, logger=logger)
                            os.remove(lp)
                        except Exception:
                            logger.exception('Error when push message or cleanup.')
                    else: # invalid string '' means blob not exist, so should delete
                        logger.error('Blob doesnot exist, will delete message %s', m.content)
                        service.deleteMessage(m, logger=logger)
                        states[num] += 1
                else: #exception, wait til next message
                    sleep(args.receipt_waiting_interval)
                    states[num] += 1
            else: # fail parse json
                # delete message
                logger.error('Bad message format, will delete message %s', m.content)
                service.deleteMessage(m, logger=logger)
                states[num] += 1
        else: # no message available
            sleep(args.receipt_waiting_interval)
            states[num] += 1



if __name__ == "__main__":
    logger = createLogger('main')
    service = createAzureService(logger)
    if service is None:
        exit(1)
    if args.mode == 'delete':
        service.cleanUp()
        exit(0)
    elif args.mode == 'upload':
        service.uploadFolder(args.imgsdir, logger)
        exit(0)
    elif args.mode == 'show':
        logger.info('azure info: %s', str(service.count()))
        exit(0)
    elif args.mode == 'process' or args.mode == 'process-local':
        manager = Manager()
        states = manager.dict()
        server = None
        serverprocess = None
        processes = []
        process_args = []
        if args.mode == 'process':
            ocrFunction = ocrQueue
        if args.mode == 'process-local':
            ocrFunction = ocrLocalPath
        def initAll():
            global server, serverprocess, processes, process_args, ocrFunction
            server = LocalServer(args.model_path, manager)
            serverprocess = Process(target=runserver, args=(server, states)) 
            processes = []
            process_args = []
            for i in range(args.numprocess):
                reader = PagePredictor(server, logger)
                states[i] = 0
                p = Process(target=ocrFunction, args=(reader, i, states))
                processes.append(p)
                process_args.append((reader, i, states))
                logger.info('new process %d with args ...', i)
            
        def startAll():
            states['server_started'] = False
            serverprocess.start()
            while not states['server_started']:
                sleep(1)
            for i in range(args.numprocess):
                processes[i].start()
        
        def killAll():
            for i in range(args.numprocess):
                processes[i].terminate()
            serverprocess.terminate()
            for i in range(args.numprocess):
                processes[i].join()
                logger.info('process %d finished', i)
            serverprocess.join()
            logger.info('server finished')
        
        initAll()
        startAll()
        
        try:
    
            oldstate = {}
            tempstate = {}
            while True:
                adding = False
                for i in range(args.numprocess):
                    if i not in oldstate:
                        oldstate[i] = states[i]
                        logger.info('add new state process %d: %d', i, oldstate[i])
                        adding = True
                    else:
                        tempstate[i] = states[i] - oldstate[i]
                        oldstate[i] = states[i]
                if not adding: 
                    if all([x==0 for x in tempstate.itervalues()]):
                        logger.error('ALL PROCESS UNCHANGE, RESTART SERVER AND ALL.')
                        killAll()
                        initAll()
                        startAll()
                    else:
                        for i in range(args.numprocess):
                            if tempstate[i] == 0:
                                logger.error('PROCESS %d UNCHANGE, NEED RESTART', i)
                                processes[i].terminate()
                                processes[i].join()
                                processes[i] = Process(target=ocrFunction, args=process_args[i])
                                processes[i].start()   
                            else:
                                logger.info('state of process %d changed: %d -> %d', i, oldstate[i]-tempstate[i], oldstate[i])                 
                sleep(args.heartbeat_check)
    
        except KeyboardInterrupt:
            logger.info('Caught KeyboardInterrupt, terminating...')
            killAll()
    
        else:
            logger.info('Quitting normally')
            for i in range(args.numprocess):
                p[i].join() 
            serverprocess.join()           
        
