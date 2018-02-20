#!/usr/bin/env python
import sys
import os
from time import sleep
from multiprocessing import Process
from pagepredictor import PagePredictor
from linepredictor import LocalServer
   
def runserver(server):
    server.run()                 
                    
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
    server = LocalServer('/home/loitg/debugtf/model_version5/')
    page_read = PagePredictor(server)

    p = Process(target=runserver, args=(server,))
    p.start()
    sleep(5000)
    
    ret = page_read.ocrImage(sys.argv[2])
    print ret   
    print(time.time() -tt)

    p.join()   