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
# from fuzzywuzzy import fuzz
from datetime import datetime, date, time

    
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

class FuzzyThenExactExtractor(object):
    def __init__(self, fuzzyregex, exactregex):
        self.fuzzyregex = fuzzyregex
        self.exactregex = exactregex
        
    def recognize(self, line):    
        m = self.fuzzyregex.recognize(line)
        if m[0] >= 0:
            n = self.exactregex.recognize(line[m[0]+len(m[1]):])
            return n
        else:
            return m

class LocodeExtractor(object):
    def __init__(self, csvdb, jarfile):
        self.csvdb = csvdb
        self.jarfile = jarfile
        
    def recognize(self, filepath):
        pass
        return ''
    
ID_KW = r'(Receipt|RECEIPT|Rcpt|Bill|BILL|CHK|Rec No|Trans|TRANS|Order|ORDER|COUNTER|Invoice|INVOICE|Serial|Check|CHECK)'
MONEY0 = ".*?\$[ ]?([1-9]\d{0,3}\.?\d{1,2})"
MONEY = ".*?(\$|S\$)?[ ]*([1-9]\d{0,3}\.\d{1,2})"
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
        print(Fore.GREEN + 'trying ' + kwtype + ' for line "' + line + '", from position ' + str(frompos))
        pos, m = recognizer.recognize(line[frompos:])
        if pos >= 0:
            print(Fore.GREEN + 'extracted-0 ' + m + ' as "' + kwtype + '"')
            self.values[kwtype].append((linenumber, float(m)))
            line = line[:frompos + pos] + ' ' + line[frompos + pos + len(m):]
        else:
            temp =  1.0*sum(c.isdigit() or c in ['$','.'] for c in nextline)
            if len(nextline) > 2:
                print(Fore.GREEN + 'next line val is ' + str(1.0*temp/len(nextline)))
            else:
                print(Fore.GREEN + 'next line val is INF.')
            if len(nextline) > 2 and 1.0*temp/len(nextline) > 0.7:
                pos, m = recognizer.recognize(nextline)
                if pos >= 0:
                    print(Fore.GREEN + 'extracted-1 ' + m + ' as "' + kwtype + '"')
                    self.values[kwtype].append((linenumber, float(m)))
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
            print(Fore.GREEN + 'trying ' + kwtype + ' for line "' + line+ '"')
            pos, m = self.id.recognize(line[frompos:])
            if pos >= 0:
                print(Fore.GREEN + 'extracted-0 ' + m + ' as "' + kwtype + '"')
                self.values[kwtype].append((linenumber, m))
                line = line[:frompos + pos] + ' ' + line[frompos + pos + len(m):]
        return line, nextline
        
                
class KWDetector(object):
    def __init__(self):
        self.types = {'total':['total', 'amount', 'payment', 'visa', 'master', 'amex', 'please pay', 'qualified amt', 'qualified amount', 'net', 'nets', 'due'],
                   'subtotal':['sub-total', 'subttl', 'payable'],
                   'cash':['cash', 'cash payment',],
                   'changedue':['change due'],
                   'nottotal':['total pts', 'total savings', 'total qty', 'total quantity', 'total item', 'total number', 'total disc', 'qty total', 'total no.', 'total direct', 'total point'],
                   'gst':['gst 7', '7 gst', 'gst', 'inclusive', 'G.S.T.'],
                   'servicecharge':['service charge', 'svr chrg', 'SVC CHG', 'SvCharge', 'Service Chg'],
                   'receiptid':['receipt', 'rcpt', 'bill', 'chk', 'trans', 'order', 'counter', 'invoice', 'serial', 'check', 'tr:']
                   }
        self.kwExtractor = KWExtractor()
        self.type_list = []
        for kwtype, kws in self.types.iteritems():
            for kw in kws:
                numwords = len(kw.split(' '))
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
        print(reg + ' ' + str((len(rawkw)+2)/6))
        return FuzzyRegexExtractor(reg, maxerr=(len(rawkw)+2)/6, caseSensitive=True)
        
    def detect(self, lines):
        for i, oriline in enumerate(lines):
            print(Fore.WHITE + oriline)
            line = oriline
            kwtypes = []
            nextline = lines[i+1] if i < len(lines) - 1 else ''
            for _, _, extr, kw, kwtype in self.type_list:
                pos, match = extr.recognize(line)
                if pos >= 0:
                    line = removeMatchFromLine((pos, match), line)
                    print(Fore.BLUE + 'match ' + match +' as "' + kwtype + '", remaining "' + line+'"')
                    kwtypes.append(kwtype)
                    line, nextline = self.kwExtractor.extract(i, kwtype, line, pos , nextline)
    
    
# TOTAL_KEYWORDS = "(total incl.of gst|check total|total|grand total|total ammount|total amt|amount|net|payment|amt payable|due|qualified amt|visa|master|grand)"
TOTAL_S = "(TOTAL|Total|[aA]mount|AMOUNT|[pP]ayment|PAYMENT|[vV]isa |VISA |MASTER |[mM]aster |AMEX |Please Pay|Qualified Amt|Qualified Amount)"
# TOTAL_I = "(grand total|qualified amt|total amt|qualified amount|total amount)"  
TOTAL_0 = "(Net[ :]|NET[ :]|NETS[ :]|Nets[ :]|Due[ :]|DUE[ :])"
NOTTOTAL = "(total pts|total saving|total qty|total quantit|total item|total number|total disc|qty total|total no\.|total direct|total point)"
TOTAL2 = "(CASH|Cash|Cash Payment|Change Due|CHANGE DUE|[pP]ayable|PAYABLE|SUB[ -]?TOTAL|Sub[ -]?Total|SUBTTL|Sub[ -]TOTAL)"



month3 = '(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)'
ddmmyy_slash = (r'(\s|^|\D)(([012]?\d|3[01])/([012]?\d|3[01])/(20)?1[78])', 2, ["%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d/%m/%y"])
yyddmm_slash = (r'(\s|^|\D)((20)?1[78]/([012]?\d|3[01])/([012]?\d|3[01]))(\s|$|\D)', 2, ["%Y/%m/%d", "%Y/%d/%m", "%y/%m/%d", "%y/%d/%m"])
ddmmyy_minus = (r'(\s|^|\D)(([012]?\d|3[01])-([012]?\d|3[01])-(20)?1[78])', 2, ["%m-%d-%Y", "%d-%m-%Y", "%m-%d-%y", "%d-%m-%y"])
yyddmm_minus = (r'(\s|^|\D)((20)?1[78]-([012]?\d|3[01])-([012]?\d|3[01]))(\s|$|\D)', 2, ["%Y-%m-%d", "%Y-%d-%m", "%y-%m-%d", "%y-%d-%m"])
ddmmyy_dot = (r'(\s|^|\D)(([012]?\d|3[01])\.([012]?\d|3[01])\.(20)?1[78])(\s|$|\D)', 2, ["%m.%d.%Y", "%d.%m.%Y", "%m.%d.%y", "%d.%m.%y"])
yyddmm_dot = (r'(\s|^|\D)((20)?1[78]\.([012]?\d|3[01])\.([012]?\d|3[01]))(\s|$|\D)', 2, ["%Y.%m.%d", "%Y.%d.%m", "%y.%m.%d", "%y.%d.%m"])
yyddmm_none = (r'(\s|^|\D)((20)?1[78]([012]\d|3[01])([012]\d|3[01]))(\s|$|\D)', 2, ["%Y%m%d", "%Y%d%m", "%y%m%d", "%y%d%m"])
ddmmyy_none = (r'(\s|^|\D)(([012]\d|3[01])([012]\d|3[01])(20)?1[78])(\s|$|\D)', 2, ["%m%d%Y", "%d%m%Y", "%m%d%y", "%d%m%y"])
ddbbyy = (r'(\s|^|\D)(([012]?\d|3[01]) ' + month3 + '[\', ]{0,2}(20)?1[78])', 2, ["%d%b%y", "%d%b%Y"])
bbddyy = (month3 + '[\', ]{0,2}([012]\d|3[01])[ ,]{0,2}(20)?1[78]', 0, ["%b%d%y", "%b%d%Y"])
IIMMSS = (r'(\s|^|\D)([01]?\d:[0-5]?\d(:[0-5]\d)?[ ]?([AP]m|[AP]M|[ap]m))', 2, ["%I:%M:%S%p", "%I:%M%p"])
HHMMSS = (r'(\s|^|\D)([012]?\d:[0-5]?\d:[0-5]?\d)(\s|$|\D)', 2, ["%H:%M:%S"])
HHMM = (r'(\s|^|\D)([012]?\d:[0-5]?\d)(\s|$|\D)', 2, ["%H:%M"])

class ReceiptIdExtractor(object):
    def __init__(self):
        pass
    
    def mostPotential(self, idlist):  
    def extract(self, lines, kwvalues):
        ids0 = kwvalues['receiptid']
        if len(ids0) > 0:
            return self.mostPotential(ids0)
        
class TotalExtractor(object):
    def __init__(self):
        pass
        
    def extract(self, lines, kwvalues):
        pass
                

class DateExtractor(object):
    def __init__(self):
        self.rawdatelist = [ddmmyy_slash, yyddmm_slash, ddmmyy_minus, yyddmm_minus, ddmmyy_dot, yyddmm_dot, yyddmm_none, ddmmyy_none, ddbbyy, bbddyy]
        self.rawtimelist = [IIMMSS, HHMMSS, HHMM]
        self.dateextrs = [RegexExtractor(x[0], x[1]) for x in self.rawdatelist]
        self.timeextrs = [RegexExtractor(x[0], x[1]) for x in self.rawtimelist]
    
    def charToNum(self, oristr):
        doubled = oristr.replace('O','0').replace('U','0').replace('D','0').replace('B','8').replace('//','7/')
        return oristr + ' ' + doubled
    
    def cleanTimeString(self, oristr):
        return oristr.replace('\'','').replace(',','').replace(' ','')
    
    def extract(self, lines):
        date_cands = []
        for i, line in enumerate(lines):
            print(Fore.WHITE + line)
            line = self.charToNum(line)
            for j, extr in enumerate(self.dateextrs):
                
                pos, cand_d_str = extr.recognize(line)
                if pos >=0:
                    cand_d_str = self.cleanTimeString(cand_d_str)
                    print(Fore.YELLOW + 'with raw string ' + cand_d_str)
                    for dateformat in self.rawdatelist[j][2]:
                        try:
                            print(Fore.YELLOW + 'trying ' + dateformat)
                            cand_d = datetime.strptime(cand_d_str, dateformat).date()
                        except Exception:
                            continue
                        today = date(2017,8,2) #datetime.today().date()
                        if cand_d <= today:
                            print(Fore.RED + str((today - cand_d).days) + ': ' + str(cand_d))
                            date_cands.append([(today - cand_d).days, cand_d, i])
        date_cands.sort()
        print('--------------------')
#         if len(date_cands) == 0:
#             return None
#         choosen_date = date_cands[0][1]
#         choosen_date_lines = [x[2] for x in date_cands if x[0]==date_cands[0][0]]
        time_cands = []
        for i, line in enumerate(lines):
            print(Fore.WHITE + line)
            line = self.charToNum(line)
            for j, extr in enumerate(self.timeextrs):
                pos, cand_t_str = extr.recognize(line)
                if pos >=0:
                    cand_t_str = self.cleanTimeString(cand_t_str)
                    print(Fore.YELLOW + 'with raw string ' + cand_t_str)
                    for timeformat in self.rawtimelist[j][2]:
                        try:
                            print(Fore.YELLOW + 'trying ' + timeformat)
                            cand_t = datetime.strptime(cand_t_str, timeformat).time()
                        except Exception:
                            continue
                        print(Fore.RED + str(cand_t))
                        time_cands.append((i,cand_t))
#         if len(time_cands) == 0:
#             return datetime.combine(choosen_date, time(0,0,0))
#         sorted_time_cands = []
#         for i, cand_t in time_cands:
#             to_chosen_date =   min([abs(i - i_cd) for i_cd in choosen_date_lines])   
#             to_chosen_date = min(to_chosen_date, 2)
#             to_chosen_date = -to_chosen_date
#             sorted_time_cands.append((to_chosen_date, cand_t))
#         sorted_time_cands.sort(reversed=True)
#         return datetime.combine(choosen_date, sorted_time_cands[0][1])
        return None
                


# ID_KW = r'(Receipt|RECEIPT|Rcpt|Bill|BILL|CHK|Rec No|Trans|TRANS|Order|ORDER|COUNTER|Invoice|INVOICE|Serial|Check|CHECK)'
# ID_VAL = r'[A-Z]{0,3}[0-9]+([-/][0-9]{1,6}([-/][0-9]+[A-Z]{0,3})?)?'
# id = ID_KW + r'[ ]?\w*?[ :\.#]{0,4}.*?' + ID_VAL
# id = RegexExtractor([id])

if __name__ == '__main__':
    allrs = {}
    with open('/tmp/temp/rs.txt','r') as f:
        fn = None
        for line in f:
            temp = line.split('----------------')
            if '.JPG----------------' in line and len(temp) > 1:
                fn = temp[0]
                allrs[fn] = []
            else:
                rs = re.findall(r'<== (.*?) == (.*?) == (.*?) == >', line)
                temp = ''
                for i in rs:
                    temp += i[0]+ ' '
                allrs[fn].append(temp)
    
    kwt = KWDetector()
    currentfile = '1501684279753_59ed9d0b-61c4-4241-9103-6f4ab2fe0684.JPG'
    for fn, lines in allrs.iteritems():
        if currentfile is not None:
            if fn != currentfile:
                continue
            else:
                currentfile = None
        kwt.detect(lines)

        print(Fore.RED + fn + ':') 
        k = raw_input("next")
    
    