'''
Created on Feb 27, 2018

@author: loitg
'''
import os
from azure.storage.blob import BlockBlobService
from azure.storage.queue import QueueService
from common import args
from receipt import ExtractedData, ReceiptSerialize
import logging

# DefaultEndpointsProtocol=https;
# AccountName=capitastarstorageacctest;AccountKey=A51dFg8jfMWTsjRSvT40GwaHxcNnUGz1bRiu6JZAuvnLBthzh+iITi6507REwLYo23ZZeCPVWY1i8zLRTxAvnQ==;
# EndpointSuffix=core.windows.net
# ------------------------
# DefaultEndpointsProtocol=http;
# AccountName=storacctcapitastartable;
# AccountKey=Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA==

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join(args.download_dir, 'log.txt'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

class AzureService(object):
    VISIBILITY_TIMEOUT = 5

    def __init__(self, account_name, account_key, container_name, queue_get, queue_push):
        self.ctnname = container_name
        self.getname = queue_get
        self.pushname = queue_push
        
        self.qs = QueueService(account_name=account_name, 
                               account_key=account_key+'l',
                               protocol='https',
#                                endpoint_suffix='core.windows.net'
                                )
        self.bs = BlockBlobService(account_name=account_name, 
                         account_key=account_key)
        
        self.qs.create_queue(self.getname)
        self.qs.create_queue(self.pushname)
        self.bs.create_container(self.ctnname)
    
    def pushMessage(self, message, qname=None):
        if qname is None:
            qname = self.pushname
        self.qs.put_message(self.pushname, message) 
        
    def getMessage(self, qname=None, num=1):
        if qname is None:
            qname = self.getname
        message = self.qs.get_messages(qname, num, visibility_timeout=self.VISIBILITY_TIMEOUT)
        return message
    
    def getReceiptInfo(self):
        message = self.getMessage()
        if len(message) > 0:
            rinfo = ReceiptSerialize.fromjson(message[0].content)
            return message[0], rinfo
        else:
            return None, None
        
    def count(self):
        metadata_get = self.qs.get_queue_metadata(self.getname)
        metadata_push = self.qs.get_queue_metadata(self.pushname)
        generator = self.bs.list_blobs(self.ctnname)
        bc = 0
        for blob in generator:
            bc += 1
        return {'get_count' : metadata_get.approximate_message_count, 
                'push_count': metadata_push.approximate_message_count,
                'blob_count': bc
                } 
    
    def uploadFolder(self, folderpath):
        for filename in os.listdir(folderpath):
            if len(filename) > 4:
                suffix = filename[-4:].upper()
            else:
                continue
            if '.JPG' == suffix or 'JPEG' == suffix:
                receipt_metadata = ReceiptSerialize()
                receipt_metadata.receiptBlobName = unicode(filename, 'utf-8')
                self.qs.put_message(self.getname, receipt_metadata.toString()) 
                self.bs.create_blob_from_path(self.ctnname, receipt_metadata.receiptBlobName, os.path.join(folderpath, filename), max_connections=2, timeout=None)
    
    def getImage(self, imgname):
        localpath= os.path.join(args.download_dir, imgname)
        self.bs.get_blob_to_path(self.ctnname, imgname, localpath)
        return localpath
    
    def deleteMessage(self, message, qname=None): 
        if qname is None:
            qname = self.getname  
        self.qs.delete_message(qname, message.id, message.pop_receipt)
     
    def deleteImage(self, imgname):   
        self.bs.delete_blob(self.ctnname, imgname)
        
    def cleanUp(self):
        count = 0
        print('deleted: ')
        while True:
            messages = self.qs.get_messages(self.getname)
            for message in messages:
                count += 1
                self.qs.delete_message(self.getname, message.id, message.pop_receipt)
            if len(messages) == 0: break
        print(str(count) + ' from queue-get')
        count = 0
        while True:
            messages = self.qs.get_messages(self.pushname)
            for message in messages:
                count += 1
                self.qs.delete_message(self.pushname, message.id, message.pop_receipt)
            if len(messages) == 0: break
        print(str(count) + ' from queue-push') 
        count = 0
        generator = self.bs.list_blobs(self.ctnname)
        for blob in generator:
            count += 1     
            self.bs.delete_blob(self.ctnname, blob.name)
        print(str(count) + ' from container') 
          
if __name__ == '__main__':
    rootLogger.debug('This message should appear on the console')
    rootLogger.info('So should this')
    rootLogger.warning('And this, too')
    rootLogger.info('Started')
    try:
        service = AzureService(account_name='storacctcapitastartable', 
                               account_key='Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA==',
                               container_name='loitg-local',
                               queue_get='loitg-queue-get',
                               queue_push='loitg-queue-push',
                               )
    except Exception as e:
        print e
#         logging.error(e)
        raise
    
    import sys
    print(sys.stdout.encoding)
    print service.count()
    
    m, rinfo = service.getReceiptInfo()
    print rinfo.toString()
    if m is not None:
        lp = service.getImage(rinfo.receiptBlobName+'t')
         
         
        import cv2
        img = cv2.imread(lp)
        cv2.imshow('dd', img)
        cv2.waitKey(-1)
        
        service.deleteMessage(m)
        service.deleteImage(rinfo.receiptBlobName)
    
    else:
        print('Empty')
    
    
    
    
    
    