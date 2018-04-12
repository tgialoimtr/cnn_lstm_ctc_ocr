'''
Created on Nov 19, 2017

@author: loitg
'''
import logging
import cv2
import ocrolib
from ocrolib.toplevel import *
from ocrolib import psegutils,morph,sl


def imgquality(lines, bounds, logger):
    rs = 0
    try:      
        bad_resolution = 0
        blur = 0
        both = 0
        total = 0
        for line, linedesc in zip(lines, bounds):
            y0,x0,y1,x1 = [int(x) for x in [linedesc[0].start,linedesc[1].start, \
              linedesc[0].stop,linedesc[1].stop]]     
            temp = cv2.Laplacian(line, cv2.CV_64F).var()
            if  temp < 700:
                blur += 1
            if y1-y0 < 15:
                bad_resolution += 1
            if temp < 1000 and y1-y0 < 20:
                both += 1
            total += 1
        if total < 4:
            print 'Discontinuous character'
            rs = rs | (1<<1)
        else:
            if 1.0*bad_resolution/total > 0.6:
    #             print bad_resolution,'/',total
                print 'Low resolution line'
                rs = rs | (1<<2)
            if 1.0*blur/total > 0.6:
    #             print blur,'/',total
                print 'Blurred'
                rs = rs | (1<<3)
            if 1.0*both/total > 0.5:
    #             print both,'/',total
                print 'Blurred and small'
                rs = rs | (1<<4)
    except:
        logging.exception("Something awful happened!")
        print 'GOOD'
        rs = 0
    return rs
        
        
if __name__ == '__main__':
    pass
        
        
        
        
        
        
        
        
        
        
        
        
        
