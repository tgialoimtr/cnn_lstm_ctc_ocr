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
from azure.common import AzureException, AzureMissingResourceHttpError
from time import sleep
from common import args

# DefaultEndpointsProtocol=https;
# AccountName=capitastarstorageacctest;AccountKey=A51dFg8jfMWTsjRSvT40GwaHxcNnUGz1bRiu6JZAuvnLBthzh+iITi6507REwLYo23ZZeCPVWY1i8zLRTxAvnQ==;
# EndpointSuffix=core.windows.net
# ------------------------
# DefaultEndpointsProtocol=http;
# AccountName=storacctcapitastartable;
# AccountKey=Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA==

class AzureService(object):
    VISIBILITY_TIMEOUT = 5

    def __init__(self, connection_string, container_name, queue_get, queue_push, logger=None):
        self.ctnname = container_name
        self.getname = queue_get
        self.pushname = queue_push
        
        self.qs = QueueService(connection_string=connection_string,
                               protocol='https',
#                                endpoint_suffix='core.windows.net'
                                )
        self.bs = BlockBlobService(connection_string=connection_string)
        self.qs.create_queue(self.getname, timeout=1)
        self.qs.create_queue(self.pushname, timeout=1)
        self.bs.create_container(self.ctnname, timeout=1)
        if logger: logger.info('Init Azure success')
    
    def pushMessage(self, message, qname=None, logger=None):
        if qname is None:
            qname = self.pushname
        try:
            self.qs.put_message(self.pushname, message) 
        except Exception as e:
            if logger:
                logger.exception('ERROR PUSH MESSAGE ')
            else:
                print 'ERROR PUSH MESSAGE '
                print e
        
    def getMessage(self, qname=None, num=1, logger=None):
        if qname is None:
            qname = self.getname
        try:
            message = self.qs.get_messages(qname, num, visibility_timeout=self.VISIBILITY_TIMEOUT)
        except Exception as e:
            if logger:
                logger.exception('ERROR GET MESSAGE ')
            else:
                print 'ERROR GET MESSAGE '
                print e
            return []
        return message
    
    def getReceiptInfo(self, logger=None):
        message = self.getMessage(logger=logger)
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
    
    def uploadFolder(self, folderpath, logger):
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
                logger.info('upload %s', filename)
    
    def getImage(self, imgname, logger=None):
        localpath= os.path.join(args.download_dir, imgname)
        try:
            self.bs.get_blob_to_path(self.ctnname, imgname, localpath)
        except AzureMissingResourceHttpError as e:
            if logger:
                logger.error('Blob named ' + imgname + ' doesnot exist.' , exec_info=True)
            else:
                print 'Blob named ' + imgname + ' doesnot exist.' 
                print e            
            return ''
        except Exception as e:
            if logger:
                logger.error('Exception while getting blob.', exec_info=True)
            else:
                print 'Exception while getting blob.' 
                print e            
            return None
        return localpath
    
    def deleteMessage(self, message, qname=None, logger=None): 
        if qname is None:
            qname = self.getname
        try:
            self.qs.delete_message(qname, message.id, message.pop_receipt)
        except Exception as e:
            if logger:
                logger.exception('ERROR DELETE MESSAGE ')
            else:
                print 'ERROR DELETE MESSAGE '
                print e
                
    def deleteImage(self, imgname, logger=None):
        try: 
            self.bs.delete_blob(self.ctnname, imgname)
        except Exception as e:
            if logger:
                logger.exception('ERROR DELETE IMAGE ')
            else:
                print 'ERROR DELETE IMAGE '
                print e
        
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
    print('started')
    try:
        service = AzureService(connection_string=args.connection_string,
                               container_name='loitg-local',
                               queue_get='loitg-queue-get',
                               queue_push='loitg-queue-push',
                               )
    except AzureException as e:
        print 'Connection Error: ' + 'Wrong credential maybe'
        print e

    import sys
    print(sys.stdout.encoding)
    print service.count()
    sleep(20)
    m, rinfo = service.getReceiptInfo()
    if m is not None: # got message
        if rinfo is not None: # parse success
            print rinfo.toString()
            lp = service.getImage(rinfo.receiptBlobName+'t')
            if lp is not None:
                if len(lp) > 0:
                    import cv2
                    img = cv2.imread(lp)
                    cv2.imshow('dd', img)
                    cv2.waitKey(-1)
                    
                    service.deleteMessage(m)
                    service.deleteImage(rinfo.receiptBlobName)
                else: # invalid string '' means blob not exist, so should delete
                    pass
            else: #exception, wait til next message
                sleep(4)
        else: # fail parse json
            # delete message
            pass
    else: # no message available
        #sleep
        print('Empty')
    
    
    
    
    
    