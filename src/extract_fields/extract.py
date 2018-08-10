'''
Created on Feb 12, 2018

@author: loitg
'''
from __future__ import print_function
import re, os
#import colorama
#from colorama import Fore
# from fuzzywuzzy import fuzz
from datetime import datetime, date, time, timedelta
from extractor import RegexExtractor, FuzzyRegexExtractor, KWDetector, ALLMONEY
from common import args
from subprocess import *
import random
from common import args
from inputoutput.receipt import ExtractedData

class LocodeExtractor(object):
    def __init__(self, csvdb, jarfile):
        self.csvdb = os.path.join(args.javapath, csvdb)
        self.jarfile = os.path.join(args.javapath, jarfile)

    def jarWrapper(self, txtfilepath):
        process = Popen(['java', '-jar', self.jarfile, self.csvdb, txtfilepath], stdout=PIPE, stderr=PIPE)
        ret = []
        while process.poll() is None:
            line = process.stdout.readline()
            if line != '' and line.endswith('\n'):
                ret.append(line[:-1])
        stdout, stderr = process.communicate()
        ret += stdout.split('\n')
        ret.remove('')
        ret = ret[0]
        if ret == 'None': ret = ''
        return ret

    def extract(self, lines, kwvalues=None):
        tempfilename = str(random.randint(1,9999999)) + '.txt'
        tempfilename = os.path.join(args.javapath, tempfilename)
        with open(tempfilename, 'w') as tempfile:
            for line in lines:
                tempfile.write(line + ' ')
        
        rs = self.jarWrapper(tempfilename)
        #delete
        os.remove(tempfilename)
        return rs


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
bbddyy = (r'(' + month3 + '[\', ]{0,2}([012]\d|3[01])[ ,]{1,2}(20)?1[78])($|[^:])', 1, ["%b%d%y", "%b%d%Y"])
IIMMSS = (r'(\s|^|\D)([01]?\d:[0-5]?\d(:[0-5]\d)?[ ]?([AP]m|[AP]M|[ap]m))', 2, ["%I:%M:%S%p", "%I:%M%p"])
HHMMSS = (r'(\s|^|\D)([012]?\d:[0-5]?\d:[0-5]?\d)(\s|$|\D)', 2, ["%H:%M:%S"])
HHMM = (r'(\s|^|\D)([012]?\d:[0-5]?\d)(\s|$|\D)', 2, ["%H:%M"])

SPECIALID1 = r'(\s|^|\D)(([012][0-9]|3[01])([012][0-9]|3[01])1[78] [0-9]{5} [0-9]{4}) [0-9]{2}:[0-9]{2}'
SPECIALID2 = r'(\s|^|\D)([012]?\d|3[01])/([012]?\d|3[01])/201[78][ ]*[012]\d:[0-5]\d[ ]*([:\w]{3,7}[ ])?[ ]*([A-Z][0-9]{6})'
SPECIALID3 = r'(\s|^|\W)T(r:|#)(\d{2,12})'

class ReceiptIdExtractor(object):
    def __init__(self):
        self.s1 = FuzzyRegexExtractor(SPECIALID1, 2, maxerr=2, caseSensitive=True)
        self.s2 = FuzzyRegexExtractor(SPECIALID2, 5, maxerr=2, caseSensitive=True)
        self.s3 = FuzzyRegexExtractor(SPECIALID3, 3, maxerr=0, caseSensitive=True)
        IDALL = r'[:#].*?([A-Z0-9]{2,25}([-/][A-Z0-9]{1,8}([-/][0-9A-Z]{1,8})?)?)'
        IDALL2 = r'([A-Z0-9]{2,25}([-/][A-Z0-9]{1,8}([-/][0-9A-Z]{1,8})?)?)'
        self.idall = RegexExtractor(IDALL, 1)
        self.idall2 = RegexExtractor(IDALL2, 2)
    
    def mostPotential(self, idlist):
        if len(idlist) > 1:
            sortedlist = [] 
            for receiptid in idlist:
                if len(receiptid) > 2:
                    numdigit = sum([c.isdigit() for c in receiptid])
                    score = 1.0*numdigit/len(receiptid)
                    if score > 0.8:
                        score = 0.8 + 1.0*len(receiptid)/15
                    sortedlist.append((score, receiptid))
            sortedlist.sort(reverse=True)
            if len(sortedlist) > 0:
                return sortedlist[0][1]
            else:
                return idlist[0]
        elif len(idlist) ==1:
            return idlist[0]
        else:
            return ''
        
    def extract(self, lines, kwvalues):
        for line in lines:
            pos, m = self.s1.recognize(line)
            if pos >= 0:
                return m
        for line in lines:
            pos, m = self.s2.recognize(line)
            if pos >= 0:
                return m
        for line in lines:
            pos, m = self.s3.recognize(line)
            if pos >= 0:
                return m
        ids0 = kwvalues['receiptid']
        ids0 = [x[1] for x in ids0 if x is not None]
        if len(ids0) > 0:
            return self.mostPotential(ids0)
        else:
            newids = []
            for line in lines:
                if all(excl not in line for excl in ['GST', 'Reg', 'REG', 'UEN', 'BUS', '$', 'ROC', 'Tel', 'TEL', 'Fax', 'FAX']):
                    pos, m = self.idall.recognize(line)
                    if pos >= 0:
                        newids.append(m)
            if len(newids) > 0:
                return self.mostPotential(newids)
            else:
                newids = []
                for line in lines:
                    if all(excl not in line for excl in ['GST', 'Reg', 'REG', 'UEN', 'BUS', '$', 'ROC', 'Tel', 'TEL', 'Fax', 'FAX']):
                        pos, m = self.idall2.recognize(line)
                        if pos >= 0:
                            newids.append(m)
                if len(newids) > 0:
                    return self.mostPotential(newids)
                else:
                    return ''
        
# import os, cv2        
# def show(imgpath):
#     print('showing' + imgpath)
#     img = cv2.imread(imgpath)
#     h, w = img.shape[:2]
#     newwidth = int(950.0 * w / h)
#     if newwidth > 300:
#         img = cv2.resize(img, (newwidth, 900))
#         cv2.imshow('kk', img)
#         cv2.waitKey(500)
#     else:
#         os.system('xdg-open ' + imgpath)
          
class TotalExtractor(object):
    '''
    in total: intotal #
    validated by gst or svc: valgstsvc
    validated by random (in line range): valrand #
    in subtotal or cash: insubtotal #
    in value range (>9 and < 200): inrange
    counts > 2 (in line range): multiple #
    '''
    def __init__(self):
        self.money = RegexExtractor(ALLMONEY, 2, -1)
    
    def findMatchedRatio(self, small, big, ratio):
        matched_str = []
        for s_str in small:
            s = float(s_str)
            bounds = ((s-0.006)*ratio, (s+0.006)*ratio)
            for d_str in big:
                d = float(d_str)
                if d > 1.0 and d > bounds[0] and d < bounds[1] and (d_str not in matched_str): 
                    matched_str.append(d_str)
        return matched_str
   
    def preprocessLines(self, lines):
        self.line_money = []
        self.nonemptylineid = []
        self.valuelist = []
        for i, line in enumerate(lines):
#             print()
#             print(str(i) + ': ', end = '')
            moneys_in_line = re.finditer(ALLMONEY, line)
            self.line_money.append([])
            for m in moneys_in_line:
                m = m.group(2)
                if len(m) > 1:
#                     print(m + ', ', end='')
                    self.line_money[i].append(m)
                    if i not in self.nonemptylineid: self.nonemptylineid.append(i)
                    self.valuelist.append(m)
        self.valuelist = [float(x) for x in self.valuelist]
        self.valuelist = [x for x in self.valuelist if x < 300]
        self.meanvalue = sum(self.valuelist)/len(self.valuelist) if len(self.valuelist) > 0 else 0
        
                    
    def similar(self, floatstr1, floatstr2):
        return abs(float(floatstr1) - float(floatstr2)) < 0.1
    
    def buildFeatures(self, kwvalues):
        linerange_multiple = +3#-3
        linerange_gstvalidate = +8#-8
        assertkwvalues = {}
        assertkwvalues['total'] = 0
        assertkwvalues['subtotal'] = 0
        assertkwvalues['cash'] = 0
        fromgst = [x[1] for x in kwvalues['gst']]
        fromsvc = [x[1] for x in kwvalues['servicecharge']]
        self.features = []
        for i, moneylist in enumerate(self.line_money):
            flatten = []
            vals = set()
            for j in range(max(0, i-linerange_multiple), min(len(self.line_money), i+linerange_multiple+1)):
                flatten += [(j, m) for m in self.line_money[j]]
            for j in range(max(0, i-linerange_gstvalidate), min(len(self.line_money), i+linerange_gstvalidate+1)):
                vals |= set(self.line_money[j])
            for current in moneylist:
                count = sum([(otherm==current) for _, otherm in flatten]) - 1
                valrand = len(self.findMatchedRatio(vals, [current], 1.0/0.06542)) > 0
                intotal = False
                for j,m in kwvalues['total']:
                    if j==i and m==current:
                        assertkwvalues['total'] += 1
                        intotal = True
                insubtotal = False
                for j,m in kwvalues['subtotal']:
                    if j==i and m==current:
                        assertkwvalues['subtotal'] += 1
                        insubtotal = True            
                for j,m in kwvalues['cash']:
                    if j==i and m==current:
                        assertkwvalues['cash'] += 1
                        insubtotal = True                 
                a = len(self.findMatchedRatio(fromgst, [current], 1.0/0.06542)) > 0
                b = len(self.findMatchedRatio(fromsvc, [current], 1.0/0.08500)) > 0
                valgstsvc = a or b
                current_val = float(current)
                self.features.append(( 1 if valgstsvc else 0,
                                    1 if intotal else 0,
                                    1 if valrand else 0,
                                    1 if (count > 0 and current_val > self.meanvalue) else 0,
                                    1 if current_val > 5 and current_val < 150 else 0,
                                    1 if insubtotal else 0,
                                    i,
                                    current))
#         print(str(assertkwvalues['total']) + '===' + str(len(kwvalues['total'])))
#         print(str(assertkwvalues['subtotal']) + '===' + str(len(kwvalues['subtotal'])))
#         print(str(assertkwvalues['cash']) + '===' + str(len(kwvalues['cash'])))
        self.features.sort(reverse=True)
    
    def extractPrice(self):
#         temp = self.features[:10]
#         for val1, intotal, valrand, multiple, inrange, insubtotal, i, m in temp:
#             print(Fore.GREEN + 'val:' + str(val1) + 
#                   ' ,intotal:' + str(intotal) + 
#                   ' ,valrand:' + str(valrand) + 
#                   ' ,multi:' + str(multiple) +
#                   ' ,inrange:' + str(inrange) + 
#                   ' ,insub:' + str(insubtotal) + 
#                   ' ,val:' + str(i) + '_' + m
#                   )
        
        if len(self.features) == 0:
            return 0.0
        elif len(self.features) == 1:
            return float(self.features[0][7])
        else:
            top = self.features[:2]
            if top[0][:6] == top[1][:6]:
                if self.similar(top[0][7], top[1][7]):
                    return float(top[0][7])
                else:
                    return 0.0
            else:
                return float(top[0][7])
            
            
    def extract(self, lines, kwvalues):
#         def printList(name, l, col):
#             print(col + name + ': ', end='')
#             for val in l:
#                 if type(val) == float:
#                     print('{:05.2f}'.format(val), end=',')
#                 else:
#                     print(val, end=',')
#             print('\n', end='')
        
        self.preprocessLines(lines)
        self.buildFeatures(kwvalues)
        return self.extractPrice()

#         totals = kwvalues['total']
#         gsts = kwvalues['gst']
#         svcs = kwvalues['servicecharge']
#         subtotals = kwvalues['subtotal']
#         cashes = kwvalues['cash']
# #         changedue = kwvalues['changedue']
#              
#         totals = [x[1] for x in totals]
#         printList('TOTAL', totals, Fore.RED)
#         totalfromgst = [float(x[1])/0.06542 for x in gsts]
#         printList('GST', totalfromgst, Fore.GREEN)
#         totalfromsvc = [float(x[1])/0.08500 for x in svcs]
#         printList('SVC', totalfromsvc, Fore.BLUE)
#          
#         cashes.sort()
#         fromcash = [x[1] for x in cashes]
#         printList('Cashes', fromcash, Fore.MAGENTA)
#         subtotals.sort()
#         subtotals = [x[1] for x in subtotals]
#         printList('Subtotal', subtotals, Fore.YELLOW)
#         show('/home/loitg/Downloads/part1/' + fn)

                    
                    
                    
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
    
    def extract(self, lines, kwvalues=None):
        date_cands = []
        for i, line in enumerate(lines):
#             print(Fore.WHITE + line)
            line = self.charToNum(line)
            for j, extr in enumerate(self.dateextrs):
                
                pos, cand_d_str = extr.recognize(line)
                if pos >=0:
                    cand_d_str = self.cleanTimeString(cand_d_str)
#                     print(Fore.YELLOW + 'with raw string ' + cand_d_str)
                    for dateformat in self.rawdatelist[j][2]:
                        try:
#                             print(Fore.YELLOW + 'trying ' + dateformat)
                            cand_d = datetime.strptime(cand_d_str, dateformat).date()
                        except Exception:
                            continue
                        today = datetime.today().date() #date(2017,8,2)
                        if cand_d <= today:
#                             print(Fore.RED + str((today - cand_d).days) + ': ' + str(cand_d))
                            date_cands.append([(today - cand_d).days, cand_d, i])
        date_cands.sort()
#         print('--------------------')
        if len(date_cands) == 0:
            return None
        choosen_date = date_cands[0][1]
        choosen_date_lines = [x[2] for x in date_cands if x[0]==date_cands[0][0]]
        time_cands = []
        for i, line in enumerate(lines):
#             print(Fore.WHITE + line)
            line = self.charToNum(line)
            temp_time = None
            for j, extr in enumerate(self.timeextrs):
                pos, cand_t_str = extr.recognize(line)
                if pos >=0:
                    cand_t_str = self.cleanTimeString(cand_t_str)
#                     print(Fore.YELLOW + 'with raw string ' + cand_t_str)
                    for timeformat in self.rawtimelist[j][2]:
                        try:
#                             print(Fore.YELLOW + 'trying ' + timeformat)
                            cand_t = datetime.strptime(cand_t_str, timeformat).time()
                        except Exception:
                            continue
#                         print(Fore.RED + str(cand_t))
                        if temp_time is None or cand_t >= temp_time:
                            temp_time = cand_t
            if temp_time is not None: 
                time_cands.append((i,temp_time))
        if len(time_cands) == 0:
            return None
        sorted_time_cands = []
        for i, cand_t in time_cands:
            to_chosen_date =   min([abs(i - i_cd) for i_cd in choosen_date_lines])   
            if to_chosen_date > 1:
                to_chosen_date = 1
            else:
                to_chosen_date = 0
            sorted_time_cands.append((to_chosen_date, cand_t))
        sorted_time_cands.sort(reverse=False)
        return datetime.combine(choosen_date, sorted_time_cands[0][1])

class CLExtractor(object):
    def __init__(self):
        self.kwt = KWDetector()
        self.date_extr = DateExtractor()
        self.total_extr = TotalExtractor()
        self.id_extr = ReceiptIdExtractor()
        self.locode_extr = LocodeExtractor(args.dbfile, args.locationnjar)
    
    def extract(self, orilines, kwvalues=None):        
        lines = orilines[:]
        self.kwt.detect(lines)
        locode0 = self.locode_extr.extract(orilines)
        locs = locode0.split('=,=')
        lines = orilines[:]
        self.kwt.detect(lines)
        datetime0 = self.date_extr.extract(lines, self.kwt.kwExtractor.values)
        if datetime0 is not None:
            if len(locs) == 5:
                if 'CALT' not in locs[0] and 'SEMBAWANG' not in locs[1] and datetime0.hour < 10:
                    datetime0 = datetime0  + timedelta(hours=12)
            datetime0 = datetime0.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            datetime0=''
        lines = orilines[:]
        self.kwt.detect(lines)
        total0 = self.total_extr.extract(lines, self.kwt.kwExtractor.values)
        lines = orilines[:]
        self.kwt.detect(lines)
        rid0 = self.id_extr.extract(lines, self.kwt.kwExtractor.values)
        if len(locs) < 5:
            extdata = ExtractedData(mallName=None, storeName=None, locationCode=None, zipcode=None, gstNo=None, 
                                    totalNumber=float(total0), receiptId=rid0, receiptDateTime=datetime0, status='INVALID')
        else:
            status = 'SUCCESS' if (total0 > 0.0 and datetime0 != '') else 'INVALID'
            extdata = ExtractedData(mallName=locs[1], storeName=locs[2], locationCode=locs[0], zipcode=locs[3], gstNo=locs[4], 
                                    totalNumber=float(total0), receiptId=rid0, receiptDateTime=datetime0, status=status)
        
        return extdata
            
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
    extractor = CLExtractor()
    currentfile = '1501682910546_dea1e329-695e-4c05-a750-f04f168b12d5.JPG'
    for fn, lines in allrs.iteritems():
        if currentfile is not None:
            if fn != currentfile:
                continue
            else:
                currentfile = None
                
        a,b,c,d = extractor.extract(lines)
        
        kwt.detect(lines)
        
        print(Fore.RED + 'rid: ' + a)
        print(Fore.RED + 'locode: ' +b)
        print(Fore.RED + 'total: ' + str(c))
        print(Fore.RED + 'date: ' +','+d) 
        print(Fore.RED + fn + ':') 
        k = raw_input("next")
    
    
