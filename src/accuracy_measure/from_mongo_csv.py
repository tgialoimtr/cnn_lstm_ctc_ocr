'''
Created on Feb 12, 2018

@author: loitg
'''
from __future__ import print_function
import cv2, re
import os
import colorama
from colorama import Fore

def show(imgpath):
    print('showing' + imgpath)
    img = cv2.imread(imgpath)
    h, w = img.shape[:2]
    newwidth = int(950.0 * w / h)
    if newwidth > 300:
        img = cv2.resize(img, (newwidth, 900))
        cv2.imshow('kk', img)
        cv2.waitKey(500)
    else:
        os.system('xdg-open ' + imgpath)

s = ','
        
if __name__ == '__main__':
    allrs = {}
    currentfile = None
    with open('/home/loitg/Downloads/v4_total.csv','r') as f:
        fn = None
        for line in f:
            temp = line.rstrip().split(',')
            if temp[0] == 'fileName': 
                continue
            else:
                if temp[5] == '0': temp[5] = ''
                allrs[temp[0]] = ([temp[3],temp[5],temp[6], temp[1]+' '+ temp[2]]) #code, price, date
    with open('/tmp/temp/measure2.csv','a') as of:
        for fn, rs in allrs.iteritems():
            if currentfile is not None:
                if fn != currentfile:
                    continue
                else:
                    currentfile = None
            i = 0
            preds = {}
            print(Fore.RED + fn)
            fn1 = '/home/loitg/ocrapp/tmp/textResult/'+ fn + '_0_8013.txt'
            fn2 = '/home/loitg/ocrapp/tmp/textResult/'+ fn + '_1_8013.txt'
            rss1 = os.stat(fn1).st_mtime
            rss2 = os.stat(fn2).st_mtime
            fntxt = fn2 if rss1 < rss2 else fn1
            with open(fntxt,'r') as rsf:
                for line in rsf:
                    print(Fore.BLACK + line)
             
            show('/home/loitg/part1/'+fn)
            if rs[0] != '': print('recognized location: '+str(rs[3]))
            if rs[1] != '': print('recognized price: '+str(rs[1]))
            realprice = raw_input('true price: (blank if correct) ')
            if realprice == '': realprice = rs[1]
            if rs[2] != '': print('recognized date: '+str(rs[2]))
            realdate = raw_input('true date: (blank if correct) ')            
            if realdate == '': realdate = rs[2]
            note = raw_input('note: ')  
            of.write(fn + s + rs[0] + s + rs[1] + s + rs[2] + realprice + s + realdate + s + note + '\n')
            of.flush()
    
    
#     for k in os.listdir('/home/loitg/part2/'):  
#         print k 
#         while True:
#             show('/home/loitg/part2/'+k,2)
#             val = raw_input('total: ')
#             command= raw_input('what next ? ')
#             if command != 'reshow':
#                 print k + ':' + val
#                 break
        
            
        
        
        
        
        
                     
                




