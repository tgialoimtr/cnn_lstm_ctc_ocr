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

def chooseClosestDate(d_str, cand_format):
    now = datetime.now(pytz.utc)
    sorted_d = []
    for date_format in cand_format:
        try:
            cand_d = datetime.strptime(d_str, date_format)
        except Exception:
            continue
        dist = abs(cand_d - d)
        sorted_d.append((dist, cand_d))
    if len(sorted_d) > 0:
        return min(sorted_d)[1]
    else:
        return None
    
    
class RegexExtractor(object):
    def __init__(self, regex, target_group=0):
        self.regex = regex
        self.target_group = target_group
        
    def recognize(self, line):
        m = re.search(self.regex, line, re.I)
        if m:
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

def FuzzyThenExactExtractor(object):
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
 
# TOTAL_KEYWORDS = "(total incl.of gst|check total|total|grand total|total ammount|total amt|amount|net|payment|amt payable|due|qualified amt|visa|master|grand)"
TOTAL_S = "(TOTAL|Total|[aA]mount|AMOUNT|[pP]ayment|PAYMENT|[pP]ayable|PAYABLE|[vV]isa |VISA |MASTER |[mM]aster )"
TOTAL_I = "(grand total|qualified amt|total amt|qualified amount|total amount|change due)"  
TOTAL_0 = "(Net | NET |NETS |Nets |Due |DUE )"
total2 = FuzzyRegexExtractor(TOTAL_S, maxerr=1, caseSensitive=True)
total1 = FuzzyRegexExtractor(TOTAL_I, maxerr=2, caseSensitive=False)
total0 = FuzzyRegexExtractor(TOTAL_0, maxerr=0, caseSensitive=True)

MONEY = "$?[1-9]\d{0-3}\.\d{1-3}[ ]"

month3 = '(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)'
ddmmyy_slash = (r'([012]?\d|3[01])/([012]?\d|3[01])/(20)?1[78]', 0, ["%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d/%m/%y"])
yyddmm_slash = (r'(20)?1[78]/([012]?\d|3[01])/([012]?\d|3[01])', 0, ["%Y/%m/%d", "%Y/%d/%m", "%Y/%m/%d", "%Y/%d/%m"])
ddmmyy_minus = (r'([012]?\d|3[01])-([012]?\d|3[01])-(20)?1[78]', 0, ["%m-%d-%Y", "%d-%m-%Y", "%m-%d-%y", "%d-%m-%y"])
yyddmm_minus = (r'(20)?1[78]-([012]?\d|3[01])-([012]?\d|3[01])', 0, ["%Y-%m-%d", "%Y-%d-%m", "%Y-%m-%d", "%Y-%d-%m"])
ddmmyy_dot = (r'([012]?\d|3[01])\.([012]?\d|3[01])\.(20)?1[78]', 0, ["%m.%d.%Y", "%d.%m.%Y", "%m.%d.%y", "%d.%m.%y"])
yyddmm_dot = (r'(20)?1[78]\.([012]?\d|3[01])\.([012]?\d|3[01])', 0, ["%Y.%m.%d", "%Y.%d.%m", "%Y.%m.%d", "%Y.%d.%m"])
yyddmm_none = (r'[ :]((20)?1[78]([012]\d|3[01])([012]\d|3[01]))', 1, ["%Y%m%d", "%Y%d%m", "%Y%m%d", "%Y%d%m"])
ddmmyy_none = (r'([012]\d|3[01])([012]\d|3[01])(20)?1[78]', 0, ["%m%d%Y", "%d%m%Y", "%m%d%y", "%d%m%y"])
ddbbyy = (r'([012]?\d|3[01]) ' + month3 + '[\', ]?(20)?1[78]', 0, ["%d%b%y", "%d%b%Y"])
bbddyy = (r'(' + month3 + ') ([012]\d|3[01])[\',] (20)?1[78]', 0, ["%b%d%y", "%b%d%Y"])
IIMMSS = (r'[01]?\d:[012345]?\d(:[012345]\d)?[ ]?([AP]m|[AP]M|[ap]m)', 0, ["%I:%M:%S%p", "%I:%M%p"])
HHMMSS = (r'[012]?\d:[012345]?\d:[012345]?\d', 0, ["%H:%M:%S"])

class DateExtractor(object):
    def __init__(self):
        self.rawdatelist = [ddmmyy_slash, yyddmm_slash, ddmmyy_minus, yyddmm_minus, ddmmyy_dot, yyddmm_dot, yyddmm_none, ddmmyy_none, ddbbyy, bbddyy]
        self.rawtimelist = [IIMMSS, HHMMSS]
        self.dateextrs = [RegexExtractor(x[0], x[1]) for x in self.rawdatelist]
        self.timeextrs = [RegexExtractor(x[0], x[1]) for x in self.rawtimelist]
        
    def extract(self, lines):
        date_cands = []
        for i, line in enumerate(lines):
            for j, extr in enumerate(self.dateextrs):
                pos, cand_d = extr.recognize(line)
                if pos >=0:
                    for dateformat in self.rawdatelist[j][2]:
                        try:
                            cand_d = datetime.strptime(cand_d, dateformat).date()
                        except Exception:
                            continue
                        today = datetime.today().date()
                        if cand_d <= today:
                            date_cands.append([(today - cand_d).days, cand_d, i])
        date_cands.sort()
        if len(date_cands) == 0:
            return None
        choosen_date = date_cands[0][1]
        choosen_date_lines = [x[2] for x in date_cands if x[0]==date_cands[0][0]]
        time_cands = []
        for i, line in enumerate(lines):
            for j, extr in enumerate(self.timeextrs):
                pos, cand_t = extr.recognize(line)
                if pos >=0:
                    for timeformat in self.rawtimelist[j][2]:
                        try:
                            cand_t = datetime.strptime(cand_t, timeformat).time()
                        except Exception:
                            continue
                        time_cands.append((i,cand_t))
        if len(time_cands) == 0:
            return datetime.combine(choosen_date, time(0,0,0))
        sorted_time_cands = []
        for i, cand_t in time_cands:
            to_chosen_date =   min([abs(i - i_cd) for i_cd in choosen_date_lines])   
            to_chosen_date = min(to_chosen_date, 2)
            to_chosen_date = -to_chosen_date
            sorted_time_cands.append((to_chosen_date, cand_t))
        sorted_time_cands.sort(reversed=True)
        return datetime.combine(choosen_date, sorted_time_cands[0][1])
                


ID_KW = r'(Receipt|RECEIPT|Rcpt|Bill|BILL|CHK|Rec No|Trans|TRANS|Order|ORDER|COUNTER|Invoice|INVOICE|Serial|Check|CHECK)'
ID_VAL = r'[A-Z]{0,3}[0-9]+([-/][0-9]{1,6}([-/][0-9]+[A-Z]{0,3})?)?'
id = ID_KW + r'[ ]?\w*?[ :\.#]{0,4}.*?' + ID_VAL
id = RegexExtractor([id])

class Extractor(object):
    def __init__(self):
        self.datetime = FuzzyRegexExtractor(datetime_regexes)
        
        
    def recognize(self, lines):
        total_txt = self.total.recognize(lines)
        
        

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
    
    for fn, lines in allrs.iteritems():
        found = 0
        for line in lines:
            prefound = found
            pos, match = id.recognize(line)
            if pos >= 0:
#                 print(str(m) + '---' + str(m[0]))
#                 print(line)
                found += 1
#             else:
#                 pos, match = total1.recognize(line)
#                 if pos >= 0:
# #                     print(str(m) + '---' + str(m[0]))    
# #                     print(line)
#                     found += 1
#                 else:
#                     pos, match = total2.recognize(line)
#                     if pos >= 0:
# #                         print(str(m) + '---' + str(m[0]))            
# #                         print(line)
#                         found += 1
            if prefound == found:
                print(Fore.WHITE + line)
            else:
                print(Fore.RED + line)
        print(Fore.RED + fn + ':' + str(found)) 
        k = raw_input("next")
    
    