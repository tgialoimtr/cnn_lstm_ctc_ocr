'''
Created on Oct 17, 2017

@author: loitg
'''
import string, re, math
from fuzzywuzzy import fuzz
import editdistance
import operator


# similarity_table = string.maketrans('5l1:08O','SIIIOBO')
char2num = string.maketrans('Oo$DBQSIl','005080511')

def standardize(unitext):
    temp = unitext.strip()
    temp = temp.encode('ascii','ignore').upper()
    temp = re.sub(' +',' ', temp)
    return temp

def standardize_loitg(unitext):
    return ' '+standardize(unitext)+' '
    
class Store(object):
    LOCATION_CODE = 'locationCode'
    STORE_NAME = 'store0'
    MALL_NAME = 'mall0'
    
    @staticmethod   
    def standardizeByName(colname, rawValue):
        if colname == Store.LOCATION_CODE:
            return ''
        elif colname == Store.MALL_NAME or colname == Store.STORE_NAME:
            return standardize(rawValue)
        else:
            return ''
    
    def __init__(self, storedict):
        self.locationCode = storedict[Store.LOCATION_CODE]
        self.storeKeyword = storedict[Store.STORE_NAME]
        self.mallKeyword = storedict[Store.MALL_NAME]
        self.storedict = storedict
        
    def getByColName(self, colname):
        return self.storedict[colname]
    
    def toString(self):
        return '%13s --%20s--%s' % (self.locationCode, self.mallKeyword, self.storeKeyword)

class Column(object):

    def __init__(self, name, e2p, swp):
        self.name = name
        self.e2p = e2p
        self.swp = swp
        self.values = {}
    
    def initAddRow(self, store):
        rawval = store.getByColName(self.name)
        if rawval == '': return
        newvals = rawval.split(' ')
        for val in newvals:
            val = Store.standardizeByName(self.name, val)
            if val in self.values:
                self.values[val].add(store)
            else:
                self.values[val] = set([store])
                
    def search0(self, lines):
        result1 = {}
        for line in lines:
            line = re.sub(' +',' ', line.strip().upper())
            if len(line) == 0: continue
            words = line.split(' ')
            liners = []
            for word in words:
                for value, stores in self.values.iteritems():
                    temp = editdistance.eval(word, value)*1.0
                    l = len(word)*1.0
                    if temp/l < 0.25:
#                         print(str(l-temp) +': '+word+' and ' + value)
                        liners.append((l-temp, stores))
            result2 = {}
            for num_char_match, stores in liners:
                if len(stores) < 20000:
                    for store in stores:
                        if store in result2:
                            result2[store] += num_char_match
                        else:
                            result2[store] = num_char_match
            for store, val in result2.iteritems():
                if store not in result1 or result1[store] < val:
                    result1[store] = val
        arranged = sorted(result1.items(), key=operator.itemgetter(1))
        return arranged
            
            
#         for value, stores in self.values.iteritems():
#             prob = 0.0
#             for word in words:
#                 temp = editdistance.eval(word, value)*1.0
#                 punish = len(value)*1.0
#                 l = max([len(word)*1.0, punish])
#                 x = (l - temp)/l - 0.75
#                 punish = punish*punish;
#                 punish = punish/(punish + self.swp)
#                 temp = (0.5*math.tanh(15*x)+0.5)*punish
#                 if temp > prob:
#                     prob = temp
# #             print(word + ' and ' + value + ' is ' + str(prob))
#             if prob > 0.6:
#                 for store in stores:
#                     if (store not in result1) or (store in result1 and result1[store] < prob):
#                         result1[store] = prob 
#         return result1           
        
    
    def search(self, items):
        result1 = {}
        if len(items) == 0: return result1
        for value, stores in self.values.iteritems():
            prob = 0.0
            for item in items:
                temp = editdistance.eval(item, value)*1.0
                temp = math.exp(-prob/self.e2p)
                if temp > prob:
                    prob = temp
            if prob > 0.6:
                for store in stores:
                    if (store not in result1) or (store in result1 and result1[store] < prob):
                        result1[store] = prob
        return result1
            
    def searchLong(self, alllines):
        result1 = {}
        allines_std = Store.standardizeByName(self.name, alllines);
        for value, stores in self.values.iteritems():
            sim = fuzz.partial_ratio(allines_std, value)*1.0
            punish = len(value)*1.0
            x = sim/100.0 - 0.75
            punish = punish - 2;
            punish = punish*punish;
            punish = punish/(punish + self.swp)
            prob = (0.5*math.tanh(15*x)+0.5)*punish
            if prob > 0.6:
                for store in stores:
                    if (store not in result1) or (store in result1 and result1[store] < prob):
                        result1[store] = prob
        return result1
    
    def printResult(self, result):
        for k, v in result.iteritems():
            print k.getByColName(self.name) + '---' + str(v)
                        