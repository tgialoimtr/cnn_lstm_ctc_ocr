'''
Created on Feb 12, 2018

@author: loitg
'''
from __future__ import print_function
from ctypes import cdll
lib1 = cdll.LoadLibrary('/usr/local/lib/libtre.so.5')
import tre
import re
import colorama
from colorama import Fore

    
class RegexExtractor(object):
    def __init__(self, regex, target_group=0, extra_group=-1):
        self.regex = regex
        self.target_group = target_group
        self.extra_group = extra_group
        
    def recognize(self, line):
        m = re.search(self.regex, line, re.I)
        if m:
            if self.extra_group >= 0:
                self.extra = m.start(self.extra_group), m.group(self.extra_group)
            return m.start(self.target_group), m.group(self.target_group)
        else:
            return -1, ''

class FuzzyRegexExtractor(object):
    def __init__(self, regex, target_group=0, maxerr=1, caseSensitive=True):
        self.regex = regex
        self.target_group = target_group
        self.fuzzyness = tre.Fuzzyness(maxerr = maxerr)
        if not caseSensitive:
            self.r = tre.compile(regex, tre.ICASE | tre.EXTENDED)
        else:
            self.r = tre.compile(regex, tre.EXTENDED)
        
    def recognize(self, line):
        m = self.r.search(line, self.fuzzyness)
        if m:
            return m.groups()[self.target_group][0], m[self.target_group]
        else:
            return -1, ''
    
MONEY0 = ".*?\$[ ]?([1-9]\d{0,3}\.?\d{1,2})"
MONEY = ".*?(\$|S\$)?[ ]*([1-9]\d{0,3}\.\d{1,2})"
ALLMONEY = "(^|\D)(([1-9]\d*|0)\.\d\d)"
GSTMONEY = "(^|\D)(\d\.([0-8]9|\d[1-8]|[1234678]0))"
GSTMONEY0 = "(^|\D)(1\d\.([0-8]9|\d[1-8]|[1234678]0))"
SVCMONEY = "(^|\D)(1?\d\.\d\d)"
ID = r'[ ]?\w*?[ :\.#]{0,4}.*?([A-Z]{0,3}[0-9]+([-/][0-9]{1,6}([-/][0-9]+[A-Z]{0,3})?)?)'

def removeMatchFromLine(m, line):
    return line[:m[0]] + ' ' + line[m[0]+len(m[1]):]


class KWExtractor(object):
    def __init__(self):
        self.money = RegexExtractor(MONEY, 2, 1)
        self.money0 = RegexExtractor(MONEY0, 1, -1)
        self.gst = RegexExtractor(GSTMONEY, 2, 1)
        self.gst0 = RegexExtractor(GSTMONEY0, 2, 1)
        self.id = RegexExtractor(ID, 1, -1)
        self.reset()
        
    def reset(self):
        self.values = {'total':[],
                   'subtotal':[],
                   'cash':[],
                   'changedue':[],
                   'nottotal':[],
                   'gst':[],
                   'servicecharge':[],
                   'receiptid':[]
                   } 
              
    def _process(self, linenumber, kwtype, line, frompos, nextline, recognizer):
#         print(Fore.GREEN + 'trying ' + kwtype + ' for line "' + line + '", from position ' + str(frompos))
        pos, m = recognizer.recognize(line[frompos:])
        if pos >= 0:
#             print(Fore.GREEN + 'extracted-0 ' + m + ' as "' + kwtype + '"')
            self.values[kwtype].append((linenumber, m))
            line = line[:frompos + pos] + ' ' + line[frompos + pos + len(m):]
        else:
            temp =  1.0*sum(c.isdigit() or c in ['$','.'] for c in nextline)
            if len(nextline) > 2 and 1.0*temp/len(nextline) > 0.5:
                pos, m = recognizer.recognize(nextline)
                if pos >= 0:
#                     print(Fore.GREEN + 'extracted-1 ' + m + ' as "' + kwtype + '"')
                    self.values[kwtype].append((linenumber, m))
        return line, nextline
                     
    def extract(self, linenumber, kwtype, line, frompos, nextline):
        if kwtype == 'gst':
            before = len(self.values[kwtype])
            line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.gst)
            after = len(self.values[kwtype])
            if before == after: #not match yet
                line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.gst0)
        if kwtype == 'servicecharge':
            line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.money)
        if kwtype == 'cash':
            line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.money)
        if kwtype == 'total' or kwtype == 'subtotal' or kwtype == 'changedue':
            before = len(self.values[kwtype])
            line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.money)
            after = len(self.values[kwtype])
            if before == after: #not match yet
                line, nextline = self._process(linenumber, kwtype, line, frompos, nextline, self.money0)
        if kwtype == 'receiptid':
#             print(Fore.GREEN + 'trying ' + kwtype + ' for line "' + line+ '"')
            pos, m = self.id.recognize(line[frompos:])
            if pos >= 0:
#                 print(Fore.GREEN + 'extracted-0 ' + m + ' as "' + kwtype + '"')
                self.values[kwtype].append((linenumber, m))
                line = line[:frompos + pos] + ' ' + line[frompos + pos + len(m):]
        return line, nextline
        
                
class KWDetector(object):
    def __init__(self):
        self.types = {'total':['total', 'amount', 'payment', 'visa', 'master', 'amex', 'please pay', 'qualified amt', 'qualified amount', 'net', 'nets', 'due'],
                   'subtotal':['sub-total', 'subttl', 'payable'],
                   'cash':['cash', 'cash payment'],
                   'changedue':['change due', 'change', 'total change'],
                   'nottotal':['total pts', 'total savings', 'total qty', 'total quantity', 'total item', 'total number', 'total disc', 'qty total', 'total no.', 'total direct', 'total point'],
                   'gst':['gst 7', '7 gst', 'gst', 'inclusive', 'G.S.T.', 'includes'],
                   'servicecharge':['service charge', 'svr chrg', 'SVC CHG', 'SvCharge', 'Service Chg', 'service tax'],
                   'receiptid':['receipt', 'rcpt', 'bill', 'chk', 'trans', 'order', 'counter', 'invoice', 'serial', 'check', 'tr:']
                   }
        self.kwExtractor = KWExtractor()
        self.type_list = []
        for kwtype, kws in self.types.iteritems():
            for kw in kws:
                numwords = len(kw.split(' '))
                if kwtype == 'gst' or kwtype == 'servicecharge': numwords += 5
                kwlen = len(kw)
                self.type_list.append((numwords, kwlen, self.kwToRegex(kw), kw, kwtype))
        self.type_list.sort(reverse=True)
                
    def kwToRegex(self, rawkw):
        reg = r'(^|\W)('
        if '-' in rawkw:
            kws = rawkw.split('-')
            sep = '[ -]?'
        else:
            kws = rawkw.split(' ')
            sep = ' '           
        for i, kw in enumerate(kws):
            kw.replace('.', '\\.?')
            if i == len(kws) - 1: sep = ''
            reg += '(' + kw.capitalize() + '|' + kw.upper() + ')' + sep
        reg += ')($|\W)'
#         print(reg + ' ' + str((len(rawkw)+2)/6))
        return FuzzyRegexExtractor(reg, maxerr=(len(rawkw)+2)/6, caseSensitive=True)
        
    def detect(self, lines):
        self.kwExtractor.reset()
        for i, oriline in enumerate(lines):
#             print(Fore.WHITE + oriline)
            line = oriline
            kwtypes = []
            nextline = lines[i+1] if i < len(lines) - 1 else ''
            for _, _, extr, kw, kwtype in self.type_list:
                pos, match = extr.recognize(line)
                if pos >= 0:
                    line = removeMatchFromLine((pos, match), line)
#                     print(Fore.BLUE + 'match ' + match +' as "' + kwtype + '", remaining "' + line+'"')
                    kwtypes.append(kwtype)
                    line, nextline = self.kwExtractor.extract(i, kwtype, line, pos , nextline)



    