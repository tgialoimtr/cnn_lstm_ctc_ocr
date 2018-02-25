#!/usr/bin/env python
import sys
import os
from time import sleep
from multiprocessing import Process, Manager, Pool
from pagepredictor import PagePredictor
from server import LocalServer
from extract_fields.extract import CLExtractor

from common import args
   
def runserver(server, states):
    server.run(states)                 

def readReceipt((reader, num)):
    print('start pushing image ' + str(num))
    extractor = CLExtractor()
    with open(args.javapath + str(num) + '.txt','a') as of:
        i = 0
        for filename in os.listdir(args.imgsdir):
            if filename[-3:].upper() == 'JPG' and hash(filename) % args.numprocess == num:
                ret = reader.ocrImage(args.imgsdir + filename)
                print filename + '-------------------------------'
                lines = []
                for line in ret.split('\n'):
                    lines.append(line.rstrip())
                    print lines[-1]
                rid, locode, total, dt = extractor.extract(lines)
                print str(i) + ',' + filename + ',' + rid + ',' + locode + ',' + '{:05.2f}'.format(total) + ',' + dt
                i += 1
                of.write(str(i) + ',' + filename + ',' + rid + ',' + locode + ',' + '{:05.2f}'.format(total) + ',' + dt + '\n')
                of.flush()
                
    return None

if __name__ == "__main__":
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

    manager = Manager()
    states = manager.dict()
    server = LocalServer(args.model_path, manager)
    p = Process(target=runserver, args=(server, states))

    readers = [PagePredictor(server) for i in range(args.numprocess)]
    
    states['server_started'] = False
    p.start()
    while not states['server_started']:
        sleep(1)   
    pool = Pool(processes=args.numprocess)
    pool.map(readReceipt, zip(readers, range(args.numprocess)))

    p.join()   