#!/usr/bin/env python
import sys
import os
from time import sleep
from multiprocessing import Process, Manager, Pool
from pagepredictor import PagePredictor
from linepredictor import LocalServer
   
def runserver(server, states):
    server.run(states)                 

def readReceipt((reader, path)):
    print('start pushing image ' + path)
    return reader.ocrImage(path)

# if __name__ == "__main__":
#     for 

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

    sys.argv = ['python', 'localhost:9000', '/home/loitg/Downloads/complex-bg/0.JPG']
    manager = Manager()
    states = manager.dict()
    server = LocalServer('/home/loitg/debugtf/model_version4_total/', manager)
    p = Process(target=runserver, args=(server, states))
        
    allreceipt = []
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/'):
        if filename[-3:] == 'JPG':
            allreceipt.append('/home/loitg/Downloads/complex-bg/' + filename)
#     random.sample(["some", "provider", "can", "be", "null"], 3)
    a = allreceipt[:2]
    readers = [PagePredictor(server) for i in range(len(a))]
    
    states['server_started'] = False
    p.start()
    while not states['server_started']:
        sleep(1)   
    pool = Pool(processes=len(a))
    ret = pool.map(readReceipt, zip(readers, a))
    
    for i in range(len(a)):
        print ret[i]
        print '---------------------' 


    p.join()   