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
 
# TOTAL_KEYWORDS = "(total incl.of gst|check total|total|grand total|total ammount|total amt|amount|net|payment|amt payable|due|qualified amt|visa|master|grand)"
TOTAL_S = "(TOTAL|Total|[aA]mount|AMOUNT|[pP]ayment|PAYMENT|[vV]isa |VISA |MASTER |[mM]aster |AMEX )"
TOTAL_I = "(grand total|qualified amt|total amt|qualified amount|total amount)"  
TOTAL_0 = "(Net[ :]|NET[ :]|NETS[ :]|Nets[ :]|Due[ :]|DUE[ :])"
NOTTOTAL = "(total pts|total saving|total qty|total quantit|total item|total number|total disc|qty total|total no\.|total direct|total point)"
TOTAL2 = "(CASH|Cash|Change Due|CHANGE DUE|[pP]ayable|PAYABLE|SUB[ -]?TOTAL|Sub[ -]?Total|SUBTTL|Sub[ -]TOTAL)"
MONEY0 = ".*?\$[ ]?([1-9]\d{0,3}\.?\d{1,2})"
MONEY = ".*?(\$|S\$)?[ ]*([1-9]\d{0,3}\.\d{1,2})"

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

class TotalExtractor(object):
    def __init__(self):
        self.total1a = FuzzyRegexExtractor(TOTAL_S, maxerr=1, caseSensitive=True)
        self.total1b = FuzzyRegexExtractor(TOTAL_I, maxerr=2, caseSensitive=False)
        self.total1c = FuzzyRegexExtractor(TOTAL_0, maxerr=0, caseSensitive=True)
        self.total2 = FuzzyRegexExtractor(TOTAL2, maxerr=1, caseSensitive=True)
        self.nottotal = FuzzyRegexExtractor(NOTTOTAL, maxerr=1, caseSensitive=False)
        self.money = RegexExtractor(MONEY, 2, 1)
        self.money0 = RegexExtractor(MONEY0, 1, -1)
        
    def extract(self, lines):
        def printMoney(line):
            n = self.money.recognize(line)
            if n[0] >= 0:
                extra = self.money.extra[1] if self.money.extra[0] >= 0 else 'None'
                print(Fore.BLUE + '=====> ' + n[1] + ' (extra:' + extra +')')
            else:
                n = self.money0.recognize(line)
                if n[0] >= 0:
                    print(Fore.BLUE + '=====> ' + n[1])
        for i, line in enumerate(lines):
            m = self.nottotal.recognize(line)
            oriline = line
            score = -1
            if m[0] >= 0:
                score = 0
                line = line[:m[0]] + line[(m[0] + len(m[1])):]
                
            m = self.total2.recognize(line)
            if m[0] >= 0:
                score = 1
                line = line[:m[0]] + line[(m[0] + len(m[1])):]
                printMoney(line)
                
            m = self.total1a.recognize(line)
            if m[0] >= 0:
                score = 2
                line = line[:m[0]] + line[(m[0] + len(m[1])):]
                printMoney(line)
            else:
                m = self.total1b.recognize(line)
                if m[0] >= 0:
                    score = 2
                    line = line[:m[0]] + line[(m[0] + len(m[1])):]
                    printMoney(line)
                else:
                    m = self.total1c.recognize(line)
                    if m[0] >= 0:
                        score = 2
                        line = line[:m[0]] + line[(m[0] + len(m[1])):]
                        printMoney(line)
            if score == -1:
                print(Fore.WHITE + oriline)
            elif score == 0:
                print(Fore.BLUE + oriline)
            elif score == 1:
                print(Fore.YELLOW + oriline)
            else:
                print(Fore.RED + oriline)
                
                

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
                


ID_KW = r'(Receipt|RECEIPT|Rcpt|Bill|BILL|CHK|Rec No|Trans|TRANS|Order|ORDER|COUNTER|Invoice|INVOICE|Serial|Check|CHECK)'
ID_VAL = r'[A-Z]{0,3}[0-9]+([-/][0-9]{1,6}([-/][0-9]+[A-Z]{0,3})?)?'
id = ID_KW + r'[ ]?\w*?[ :\.#]{0,4}.*?' + ID_VAL
id = RegexExtractor([id])

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
    
    total = DateExtractor()
    currentfile = '1501685133708_76911917-c833-4319-b432-4afacef5fed6.JPG'
    for fn, lines in allrs.iteritems():
        if currentfile is not None:
            if fn != currentfile:
                continue
            else:
                currentfile = None
        
        total.extract(lines)

        print(Fore.RED + fn + ':') 
        k = raw_input("next")
    
    