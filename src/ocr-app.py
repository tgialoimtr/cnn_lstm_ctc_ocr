#!/usr/bin/env python
import sys
import os
from time import sleep
from multiprocessing import Process, Manager, Pool
import logging
from logging.handlers import TimedRotatingFileHandler

from processing.pagepredictor import PagePredictor
from processing.server import LocalServer
from extract_fields.extract import CLExtractor
from inputoutput.receipt import ExtractedData, ReceiptSerialize
from common import args

def createLogger(name):
    logFormatter = logging.Formatter("%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(name)
    
    fileHandler = TimedRotatingFileHandler(os.path.join(args.javapath, 'log.' + name) , when='midnight', interval=2, backupCount=5)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger)
    
def ocrQueue(reader, num, states):
    logger = createLogger('worker-' + str(num))
    logger.info('process %d start pushing image.', num)
    extractor = CLExtractor()
#     try:
#         service = AzureService(connection_string=args.connection_string,
#                                container_name='loitg-local',
#                                queue_get='loitg-queue-get',
#                                queue_push='loitg-queue-push',
#                                )
#     except AzureException as e:
#         print 'Connection Error: ' + 'Wrong credential maybe'
#         print e

#     print service.count()
    with open(args.javapath + str(num) + '.txt','a') as of:
        while True:
            extdata = None
            try:
                lines = reader.ocrImage('/home/loitg/Downloads/complex-bg/22.JPG', logger)
                logger.info('process %d has %d lines.',num, len(lines))
                for line in lines:
                    logger.debug(line)
                rid, locode, total, dt = extractor.extract(lines)
            except Exception:
                logger.exception('EXCEPTION WHILE EXTRACTING LINES AND FIELDS.')
                extdata = ExtractedData()
            if extdata is None:
                extdata = ExtractedData(locationCode = locode, receiptId=rid, totalNumber=total, receiptDateTime=dt, status='SUCCESS')
                states[num] += 1
            meta = ReceiptSerialize()
            outmsg = str(meta.combineExtractedData(extdata))
            logger.info('%d, %s', states[num], outmsg)
            of.write(str(states[num]) + ',' + outmsg + '\n')
            of.flush()
            
            
#             m, rinfo = service.getReceiptInfo()
#             if m is not None: # got message
#                 if rinfo is not None: # parse success
#                     print rinfo.toString()
#                     lp = service.getImage(rinfo.receiptBlobName+'t')
#                     if lp is not None:
#                         if len(lp) > 0:
#                             import cv2
#                             img = cv2.imread(lp)
#                             cv2.imshow('dd', img)
#                             cv2.waitKey(-1)
#                             
#                             lines = reader.ocrImage(lp)
#                             rid, locode, total, dt = extractor.extract(lines)
#                             print str(i) + ',' + filename + ',' + rid + ',' + locode + ',' + '{:05.2f}'.format(total) + ',' + dt
#                             i += 1
#                             of.write(str(i) + ',' + filename + ',' + rid + ',' + locode + ',' + '{:05.2f}'.format(total) + ',' + dt + '\n')
#                             of.flush()
#                                               
#                             service.deleteMessage(m)
#                             service.deleteImage(rinfo.receiptBlobName)
#                         else: # invalid string '' means blob not exist, so should delete
#                             print 'Blob doesnot exist, will delete message ' + m.content
#                             service.deleteMessage(m)
#                     else: #exception, wait til next message
#                         sleep(args.receipt_waiting_interval)
#                 else: # fail parse json
#                     # delete message
#                     print 'Bad message format, will delete message ' + m.content
#                     service.deleteMessage(m)
#             else: # no message available
#                 sleep(args.receipt_waiting_interval)



if __name__ == "__main__":
    logger = createLogger('main')
    manager = Manager()
    states = manager.dict()
    server = LocalServer(args.model_path, manager)
    serverprocess = Process(target=runserver, args=(server, states)) 

    processes = []
    process_args = []
    for i in range(args.numprocess):
        reader = PagePredictor(server, logger)
        states[i] = 0
        p = Process(target=ocrQueue, args=(reader, i, states))
        processes.append(p)
        process_args.append((reader, i, states))
        logger.info('new process %d with args ...', i)
        

    states['server_started'] = False
    serverprocess.start()
    while not states['server_started']:
        sleep(1)
    for i in range(args.numprocess):
        p.start()

    try:
        oldstate = {}
        while True:
            for i in range(args.numprocess):
                if i not in oldstate:
                    oldstate[i] = states[i]
                    logger.info('add new state process %d: %d', i, oldstate[i])
                else:
                    if oldstate[i] == states[i]:
                        logger.error('PROCESS %d UNCHANGE, NEED RESTART', oldstate[i])
                        #restart
#                         processes[i].terminate()
#                         processes[i].join()
#                         processes[i] = Process(target=ocrQueue, args=(reader, i, states))
#                         processes[i].start()
                    else:
                        logger.info('state of process %d changed: %d -> %d', i, oldstate[i], states[i])
                        oldstate[i] = states[i]
            sleep(10*args.time_per_receipt)

    except KeyboardInterrupt:
        logger.info('Caught KeyboardInterrupt, terminating...')
        for i in range(args.numprocess):
            processes[i].terminate()
        serverprocess.terminate()
        for i in range(args.numprocess):
            processes[i].join()
            logger.info('process %d finished', i)
        serverprocess.join()
        logger.info('server finished')

    else:
        logger.info('Quitting normally')
        for i in range(args.numprocess):
            p[i].join() 
        serverprocess.join()           


# if __name__ == "__main__":
#     pp = PagePredictor('localhost:9000')
#     with open('/tmp/temp_hope/rs.txt', 'w') as rs:
#         for filename in os.listdir('/home/loitg/Downloads/complex-bg/'):        
#             if filename[-3:].upper() == 'JPG':
#                 
#                 tt = time.time()
#                 ret = pp.ocrImage('/home/loitg/Downloads/complex-bg/' + filename)
#                 rs.write(filename + '----------------' + str(time.time() - tt) + '\n')
#                 rs.write(ret+ '\n')
#                 rs.flush()

#     manager = Manager()
#     states = manager.dict()
#     server = LocalServer(args.model_path, manager)
#     p = Process(target=runserver, args=(server, states))
# 
#     readers = [PagePredictor(server) for i in range(args.numprocess)]
#     
#     states['server_started'] = False
#     p.start()
#     while not states['server_started']:
#         sleep(1)   
#     pool = Pool(processes=args.numprocess)
#     pool.map(readReceipt, zip(readers, range(args.numprocess)))
# 
#     p.join()   