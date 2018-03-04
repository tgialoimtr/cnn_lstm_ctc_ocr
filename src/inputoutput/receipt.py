'''
Created on Feb 27, 2018

@author: loitg
'''
import decimal
import simplejson as json

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
        
class ReceiptSerialize(object):
    '''
    For serialize
    '''
    
    def __init__(self):
        self.memberNumber = None;
        self.token = u"";
        self.amount = 1.0/3;
        self.currency = u"hhh\u2026hhh";
        self.program = None;
        self.mobileVersion = "fdfd";
        self.deviceName = None;
        self.receiptBlobName = u"";
        self.receiptCrmName = u"";
        self.station = u"";
 

    @classmethod
    def fromjson(cls, jsonStr):
        rs = cls()
        try:
            fromjson = json.loads(jsonStr, parse_float=decimal.Decimal)
        except ValueError:
            return None
        rs.memberNumber = fromjson['memberNumber'];
        rs.token = fromjson['token'];
        rs.amount = fromjson['amount'];
        rs.currency = fromjson['currency'];
        rs.program = fromjson['program'];
        rs.mobileVersion = fromjson['mobileVersion'];
        rs.deviceName = fromjson['deviceName'];
        rs.receiptBlobName = fromjson['receiptBlobName'];
        rs.receiptCrmName = fromjson['receiptCrmName'];
        rs.station = fromjson['station'];
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
        