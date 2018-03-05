'''
Created on Feb 27, 2018

@author: loitg
'''
import decimal
import simplejson as json
from base64 import b64decode

class ExtractedData(object):
    def __init__(self, mallName=None, storeName=None, locationCode=None, zipcode=None, gstNo=None, totalNumber=0.0, receiptId=None, receiptDateTime=None, status='FAIL'):
        self.mallName = mallName
        self.storeName = storeName
        self.locationCode = locationCode
        self.zipcode = zipcode
        self.gstNo = gstNo
        self.totalNumber = totalNumber
        self.receiptId = receiptId
        self.receiptDateTime = receiptDateTime
        self.status = status
        self.qualityCode = 0
        self.message = 'Receipt OK'
        
class ReceiptSerialize(object):
    '''
    For serialize
    '''
    
    def __init__(self):
        self.memberNumber = None;
        self.token = None;
        self.amount = 0.0;
        self.currency = None;
        self.program = None;
        self.station = None;
        self.mobileVersion = None;
        self.deviceName = None;
        self.receiptBlobName = None;
        self.receiptCrmName = None;
        self.uploadLocalFolder = None;
 

    @classmethod
    def fromjson(cls, jsonStr):
        rs = cls()
        try:
            try:
                fromjson = json.loads(jsonStr, parse_float=decimal.Decimal)
            except Exception:
                fromjson = json.loads(b64decode(jsonStr), parse_float=decimal.Decimal)
            rs.memberNumber = fromjson['memberNumber'];
            rs.token = fromjson['token'];
            rs.amount = fromjson['amount'];
            rs.currency = fromjson['currency'];
            rs.program = fromjson['program'];
            rs.station = fromjson['station'];
            rs.mobileVersion = fromjson['mobileVersion'];
            rs.deviceName = fromjson['deviceName'];
            rs.receiptBlobName = fromjson['receiptBlobName'];
            rs.receiptCrmName = fromjson['receiptCrmName'];
            rs.uploadLocalFolder = fromjson['uploadLocalFolder'];
        except Exception:
            return None
        return rs
    
    def toString(self):
        return json.dumps(self.__dict__).decode('utf-8')
    
    def combineExtractedData(self, extdata):
        self.mallName = extdata.mallName
        self.storeName = extdata.storeName
        self.locationCode = extdata.locationCode
        self.zipcode = extdata.zipcode
        self.gstNo = extdata.gstNo
        self.totalNumber = extdata.totalNumber
        self.receiptId = extdata.receiptId
        self.receiptDateTime = extdata.receiptDateTime
        self.status = extdata.status
        return json.dumps(self.__dict__).decode('utf-8')
        
