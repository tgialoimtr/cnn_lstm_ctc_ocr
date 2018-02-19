'''
Created on Feb 13, 2018

@author: loitg
'''

from main import s

def regex_fix(ocrlines):
    return ocrlines[0]
    
if __name__ == '__main__':
    with open('/tmp/temp/measure.csv','r') as f:
        fn = None
        allrs = {}
        allscore = [0,0,0]
        for line in f:
            temp = line.split(s)
            fn = temp[0]; bad = temp[1]; linetype=temp[2]; sol=temp[3]; gt=temp[4]; preds=(temp[5], temp[6],temp[7]);
            pred = regex_fix(preds)
            if fn not in allrs:
                allrs[fn] = {}
                allrs[fn][0] = [] #date
                allrs[fn][1] = [] #price
            if linetype == 'd':
                score = 1 if gt.upper().strip() in pred.upper() else 0
                allrs[fn][0].append(score)
            elif linetype == 'b':
                gt_price = gt.split(',')[1]
                score = 1 if gt.upper().strip() in pred.upper() else 0
                allrs[fn][1].append(score)
            elif linetype == 'p':
                score = 1 if gt.upper().strip() in pred.upper() else 0
                allrs[fn][1].append(score)
#             print sol
        for fn, dateandprices in allrs.iteritems():
            score_date = max(dateandprices[0])
            score_price = 1.0*sum(dateandprices[1])/len(dateandprices[1])
            score = score_date * score_price
            allscore[0] += score_date
            allscore[1] += score_price
            allscore[2] += score
        
        print('num receipts: ' + str(len(allrs)))
        print('score_date: ' + str(1.0*allscore[0]/len(allrs)))
        print('score_price: ' + str(1.0*allscore[1]/len(allrs)))
        print('score: ' + str(1.0*allscore[2]/len(allrs)))
        
        
if __name__ == '__main__':
    with open('/tmp/temp/measure2.csv','r') as f:
        fn = None
        allrs = {}
        allscore = {'all':0, 'p':0, 'd':0, 'l':0, 'a':0}
        for line in f:  
            temp = line.split(s)
            fn = temp[0]; p_locode = temp[1]; p_price = temp[2]; p_date=temp[3]; r_price=temp[4]; r_date=temp[5]
            pc,dc,lc = False,False,False
            if p_price != '' and r_price != '' and abs(float(p_price) - float(r_price)) < 0.5:
                allscore['p'] += 1; pc = True
            if r_date.upper().strip() in p_date.upper():
                allscore['d'] += 1; dc = True
            if p_locode != '':
                allscore['l'] += 1; lc = True
            if pc and dc and lc: allscore['a'] += 1 
            allscore['all'] += 1    
        print('num receipts: ' + str(allscore['all']))
        print('score_date: ' + str(1.0*allscore['d']/allscore['all']))
        print('score_price: ' + str(1.0*allscore['p']/allscore['all']))
        print('score_location: ' + str(1.0*allscore['l']/allscore['all']))
        print('score: ' + str(1.0*allscore['a']/allscore['all']))    
            
        