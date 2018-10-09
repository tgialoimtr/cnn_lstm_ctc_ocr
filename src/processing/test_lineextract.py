#!/usr/bin/env python
import sys
sys.path += ['/usr/local/lib/python2.7/dist-packages/mininet-2.2.1-py2.7.egg', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/loitg/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0']
import cv2
from pylab import *
from scipy.ndimage import morphology
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, uniform_filter1d
import ocrolib
from skimage.filters import threshold_sauvola

from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import time
import datetime

from linepredictor import BatchLinePredictor
from imgquality import imgquality
from test_charboxfind import findBox
import random
       
class obj:
    def __init__(self):
        pass
    

args = obj()
args.zoom = 0.5
args.range = 20
args.debug = 1
args.perc= 80
args.escale = 1.0
args.threshold = 0.5
args.lo = 5
args.hi = 90
args.usegauss = False
args.vscale = 1.0
args.hscale = 1.0
args.threshold = 0.25
args.pad = 1
args.expand = 3
args.model = '/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz'
args.inputdir = '/root/ocrapp/tmp/cleanResult/'
args.connect = 1
args.noise = 8

def rotate(image, angle):
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
    dst = cv2.warpAffine(image, M, image.shape)
    return dst
def resizeToHeight(img, newheight):
    newwidth = int(1.0*img.shape[1]/img.shape[0]*newheight)
    newimg = cv2.resize(img, (newwidth, newheight))
    return newimg
def summarize(a):
    b = a.ravel()
    return len(b),[amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])
   
def sauvola(grayimg, w=51, k=0.2, scaledown=None, reverse=False):
    mask =None
    if scaledown is not None:
        mask = cv2.resize(grayimg,None,fx=scaledown,fy=scaledown)
        w = int(w * scaledown)
        if w % 2 == 0: w += 1
        if w == 1: w=3
        mask = threshold_sauvola(mask, w, k)
        mask = cv2.resize(mask,(grayimg.shape[1],grayimg.shape[0]),fx=scaledown,fy=scaledown)
    else:
        if w % 2 == 0: w += 1
        if w == 1: w=3
        mask = threshold_sauvola(grayimg, w, k)
    if reverse:
        return where(grayimg > mask, uint8(0), uint8(1))
    else:
        return where(grayimg > mask, uint8(1), uint8(0)) 

def estimate_skew_angle(image,angles):
    estimates = []
    binimage = sauvola(image, 11, 0.1).astype(float)
    for a in angles:
        rotM = cv2.getRotationMatrix2D((binimage.shape[1]/2,binimage.shape[0]/2),a,1)
        rotated = cv2.warpAffine(binimage,rotM,(binimage.shape[1],binimage.shape[0]))
        v = mean(rotated,axis=1)
        d = [abs(v[i] - v[i-1]) for i in range(1,len(v))]
        d = var(d)
        estimates.append((d,a))
#     if args.debug>0:
#         plot([y for x,y in estimates],[x for x,y in estimates])
#         ginput(1,args.debug)
    _,a = max(estimates)
    return a


def calc_line(oriline):
    line = sauvola(oriline,w=oriline.shape[0]/2, k=0.05, reverse=True)
    oridense = '{:3.3f}'.format(mean(oriline))
    dens = '{:3.3f}'.format(mean(line))
#     rati = '{:3.3f}'.format(1.0*line.shape[1]/line.shape[0])
    _,n = morph.label(line)
    n = '{:3.3f}'.format(1.0*n/oriline.shape[1]*oriline.shape[0])
    return dens+'_'+n+'_'+oridense, line

def pre_check_line(oriline):
    if oriline.shape[0] < 10:
        return False
    if 1.0*oriline.shape[1]/oriline.shape[0] < 1.28:
        return False
    if mean(oriline) < 0.35:
        return False
    if oriline.shape[0] > 25:
        line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True, scaledown=20.0/oriline.shape[0])
    else:
        line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True)
    if mean(line) < 0.15:
        return False
    _,n = morph.label(line)
    n = 1.0*n/oriline.shape[1]*oriline.shape[0]
    if n > 15:
        return False
    return True


class CandChar(object):
    def __init__(self, bound):
        self.bound = bound
        self.lineid = None
        self.height = bound[0].stop - bound[0].start
        self.x = (bound[1].start + bound[1].stop)/2
        self.bottom_y = bound[0].stop
        self.top_y = bound[0].start

class SubLine(object):
    def __init__(self):
        self.available = True
    
    
    def mergeRightChar(self, char):
        
        return False
    
    
    def mergeRightLine(self, line):
        
        
        return False

class SubLineFinder(object):
    angle_list = np.linspace(-10,10,20)
    
    def __init__(self, window_size, cellbound, initChar = None):
        self.chars = []
        self.cellbound = cellbound
        self.window_size = window_size
        if initChar is not None:
            self.addChar(initChar)
        self.sin = {}; self.cos = {}
        for a in self.angle_list:
            self.sin[a] = math.sin(a/180.0*math.pi)
            self.cos[a] = math.cos(a/180.0*math.pi)
    
    def addChar(self, cand_char):
        if type(cand_char) is not CandChar:
            cand_char = CandChar(cand_char)
        self.chars.append(cand_char)
    
    def _findDense(self, l, windows_size):
        rets = []
        l.sort()
        for i, v in enumerate(l):
#             print 'i', i, l[i]
            if i >= len(l) - 1: continue
            total = [l[i]]
            for j in range(i+1, len(l)):
#                 print 'j', j, l[j]
                if l[j] - l[i] < windows_size:
                    total.append(l[j])
#                     print 'accum ', support, total
                    if j == len(l) - 1 and len(total) >= 3:
                        rets.append((len(total), l[i]-l[j], 1.0*sum(total)/len(total)))
                elif len(total) >= 3:
#                     print 'j now is ', l[j], ', over window of ', l[i]
                    rets.append((len(total), l[i]-l[j], 1.0*sum(total)/len(total)))
                    break
        rets.sort(reverse=True)
        return rets[0] if len(rets) > 0 else None
        
    
    def subline(self):
        if len(self.chars) == 0: return None
        rets = []
        
        for a in self.angle_list:
            topys = []
            bottomys = []
            for char in self.chars:
                newytop = char.top_y * self.cos[a] + char.x * self.sin[a]
                newybottom = char.bottom_y * self.cos[a] + char.x * self.sin[a]
                topys.append(newytop)
                bottomys.append(newybottom)
            top_pos = self._findDense(topys, self.window_size)
#             print topys, top_pos
            bottom_pos = self._findDense(bottomys, self.window_size)
#             print bottomys, bottom_pos
            if top_pos is not None and bottom_pos is not None:
                if top_pos[0] >= 3 and bottom_pos[0] >= 5:
                    rets.append((bottom_pos[1], bottom_pos[0], bottom_pos[2], a))

#         illu = np.zeros((self.cellbound[0].stop - self.cellbound[0].start, self.cellbound[1].stop - self.cellbound[1].start), dtype=np.uint8)
#         print 'fdfd', illu.shape
#         for char in self.chars:
#             x = char.x - self.cellbound[1].start
#             top_y = char.top_y - self.cellbound[0].start
#             bottom_y = char.bottom_y - self.cellbound[0].start
#             print x,top_y, bottom_y
#             cv2.line(illu, (x,top_y), (x,bottom_y), 255, 2)
#         cv2.imshow('uug', illu)
#         cv2.waitKey(-1)
        
        rets.sort(reverse=True)
        if len(rets) > 0:
            xmin = self.cellbound[1].start
            xmax = self.cellbound[1].stop
            alpha = rets[0][3];
            pos2 = rets[0][2]
#             print pos2 - self.cellbound[0].start
            ymin = -self.sin[alpha]*xmin/self.cos[alpha] + pos2/self.cos[alpha]
            ymax = -self.sin[alpha]*xmax/self.cos[alpha] + pos2/self.cos[alpha]
            return (xmin, ymin), (xmax, ymax)
        else:
            return None

def extractLines(imgpath, param):
    img_grey = ocrolib.read_image_gray(imgpath)
    (h, w) = img_grey.shape[:2]
    img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
    angle = estimate_skew_angle(img00,linspace(-5,5,42))
    print 'goc', angle

    rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
    
    h,w = img_grey.shape
    img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.999, cv2.NORM_MINMAX)
    binary = sauvola(img_grey, w=param.w, k=param.k, scaledown=0.2, reverse=True)### PARAM
    binary = morph.r_closing(binary.astype(bool), (args.connect,1))
    binaryary = binary[h/4:3*h/4,w/4:3*w/4]
    binary = binary.astype(np.uint8)
    labels,n = morph.label(binaryary)
    objects = morph.find_objects(labels)
    
    
    bysize = sorted(objects,key=sl.area)
    scalemap = zeros(binaryary.shape)
    for o in bysize:
        if amax(scalemap[o])>0: continue
        scalemap[o] = sl.area(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)]) 
    objects = psegutils.binary_objects(binary)
    boxmap = zeros(binary.shape,dtype=np.uint8)
    
    imgwidth = binary.shape[1]
    imgheight = binary.shape[0]
    cellwidth = 6*scale
    cellheight = 2.5*scale
    N_x = int(round(imgwidth / cellwidth))
    cellwidth = int(round(imgwidth / N_x))
    N_y = int(round(imgheight / cellheight))
    cellheight = int(round(imgheight / N_y))
    cells_list = [{},{},{},{}]
    def pixel2cell2id(pixel_x, pixel_y, CELLTYPE):
        dx = 0; dy = 0;
        if CELLTYPE == 3: pixel_x -= cellwidth/2; pixel_y -=cellheight/2; dx = cellwidth/2; dy = cellheight/2;
        if CELLTYPE == 2: pixel_x -= cellwidth/2; dx = cellwidth/2;
        if CELLTYPE == 1: pixel_y -= cellheight/2; dy = cellheight/2;
        if pixel_x <= 0 or pixel_y <=0: return None, None
        cellcoord = (pixel_x / cellwidth, pixel_y / cellheight)
        cellid = cellcoord[0] + cellcoord[1]*N_x
        cellcoord = (cellcoord[0]* cellwidth + dx, cellcoord[1]* cellheight + dy)
        return cellcoord, cellid
    
    def id2cell2pixel(cellid, x, y, CELLTYPE):
        cellcoord = (cellid % N_x, cellid / N_x)
        pixel_x = cellcoord[0] * cellwidth + x
        pixel_y = cellcoord[1] * cellheight + y
        if CELLTYPE == 3: pixel_x += cellwidth/2; pixel_y +=cellheight/2;
        return cellcoord, pixel_x, pixel_y
    
    img_grey = (cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)




    for o in objects:
        h = sl.dim0(o)
        w = sl.dim1(o)
        ratio = float(w)/h
        ### Dirty cheat
        if ratio > 1 and ratio < 6:
            recommended_width = max(int(0.6*(o[0].stop - o[0].start)), int(scale*0.6), 5)
            for pos in range(o[1].start +recommended_width, o[1].stop, recommended_width):
                binary[o[0].start:o[0].stop, pos:pos+1] = np.uint8(0)
    objects = psegutils.binary_objects(binary)
    
    for o in objects:
        h = sl.dim0(o)
        w = sl.dim1(o)
        a = h*w
#         black = float(sum(binary[o]))/a
#         if sl.area(o)**.5<threshold[0]*scale: continue
#         if sl.area(o)**.5>threshold[1]*scale: continue
        if h > 5*scale: continue
#         if h < 0.4*scale: continue
        if w > 4*scale and (h > 2*scale or h < 0.5*scale): continue
        if a < 0.25*scale*scale: continue
        if float(h)/w > 10:continue
        ratio = float(w)/h
        if ratio > 10: continue

        ### Add object as candidate character
        pixel_x, pixel_y = (o[1].start + o[1].stop)/2, o[0].stop
        for celltype in range(4):
            cellcoord, cellid = pixel2cell2id(pixel_x, pixel_y, CELLTYPE=celltype)
            if cellcoord is None or cellid is None: continue
            cellbound = slice(cellcoord[1], cellcoord[1] + cellheight, None), slice(cellcoord[0], cellcoord[0] + cellwidth, None)
            if cellid not in cells_list[celltype]:
                cells_list[celltype][cellid] = SubLineFinder(window_size=max(3, scale/6), cellbound=cellbound, initChar=o)
            else:
                cells_list[celltype][cellid].addChar(o)
        
        y0 = o[0].start
        y1 = o[0].stop - 3 if o[0].stop - o[0].start > 8 else o[0].start + 5
        x0 = o[1].start
        x1 = o[1].stop - 3 if o[1].stop - o[1].start > 8 else o[1].start + 5
        boxmap[y0:y1,x0:x1] = 1
        
    for celltype in range(4):
        if celltype == 0: col = (255,0,0)
        if celltype == 1: col = (0,255,0)
        if celltype == 2: col = (255,255,0)
        if celltype == 3: col = (0,0,255)
        for cellid, subline in cells_list[celltype].iteritems():
#             cv2.rectangle(img_grey, (subline.cellbound[1].start+celltype, subline.cellbound[0].start+celltype), (subline.cellbound[1].stop+celltype, subline.cellbound[0].stop+celltype), col,1)
            line = subline.subline()
            if line is not None:
                pos1 = (int(line[0][0]), int(line[0][1]))
                pos2 = (int(line[1][0]), int(line[1][1]))
#                 print cellid, pos1, pos2
                cv2.line(img_grey, pos1, pos2, col, 1)
    ### illustrate/debug first round
    
    return binary, cv2.add(img_grey, (boxmap[:,:,np.newaxis]*np.array([0,50,50])).astype(np.uint8))


def extractLines2(imgpath):
    img_grey = ocrolib.read_image_gray(imgpath)
    img_grey = img_grey[:img_grey.shape[0]/2,:]
    
    (h, w) = img_grey.shape[:2]
    img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
    angle = estimate_skew_angle(img00,linspace(-5,5,42))
    print 'goc', angle
    rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
    
    h,w = img_grey.shape
    img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.999, cv2.NORM_MINMAX)
    
    objects, scale = findBox(img_grey)
    
    imgwidth = img_grey.shape[1]
    imgheight = img_grey.shape[0]
    cellwidth = 6*scale
    cellheight = 2.5*scale
    N_x = int(round(imgwidth / cellwidth))
    cellwidth = int(round(imgwidth / N_x))
    N_y = int(round(imgheight / cellheight))
    cellheight = int(round(imgheight / N_y))
    cells_list = [{},{},{},{}]
    def pixel2cell2id(pixel_x, pixel_y, CELLTYPE):
        dx = 0; dy = 0;
        if CELLTYPE == 3: pixel_x -= cellwidth/2; pixel_y -=cellheight/2; dx = cellwidth/2; dy = cellheight/2;
        if CELLTYPE == 2: pixel_x -= cellwidth/2; dx = cellwidth/2;
        if CELLTYPE == 1: pixel_y -= cellheight/2; dy = cellheight/2;
        if pixel_x <= 0 or pixel_y <=0: return None, None
        cellcoord = (pixel_x / cellwidth, pixel_y / cellheight)
        cellid = cellcoord[0] + cellcoord[1]*N_x
        cellcoord = (cellcoord[0]* cellwidth + dx, cellcoord[1]* cellheight + dy)
        return cellcoord, cellid
    
    def id2cell2pixel(cellid, x, y, CELLTYPE):
        cellcoord = (cellid % N_x, cellid / N_x)
        pixel_x = cellcoord[0] * cellwidth + x
        pixel_y = cellcoord[1] * cellheight + y
        if CELLTYPE == 3: pixel_x += cellwidth/2; pixel_y +=cellheight/2;
        return cellcoord, pixel_x, pixel_y
    
    illu = cv2.cvtColor(img_grey.astype(np.float32), cv2.COLOR_GRAY2BGR)
    illu = cv2.resize(illu, None, fx=2.0, fy=2.0)
    illu = (illu*255).astype(np.uint8)
    
    for o in objects:
        ### Add object as candidate character
        pixel_x, pixel_y = (o[1].start + o[1].stop)/2, o[0].stop
        for celltype in range(4):
            cellcoord, cellid = pixel2cell2id(pixel_x, pixel_y, CELLTYPE=celltype)
            if cellcoord is None or cellid is None: continue
            cellbound = slice(cellcoord[1], cellcoord[1] + cellheight, None), slice(cellcoord[0], cellcoord[0] + cellwidth, None)
            if cellid not in cells_list[celltype]:
                cells_list[celltype][cellid] = SubLineFinder(window_size=scale/3, cellbound=cellbound, initChar=o)
            else:
                cells_list[celltype][cellid].addChar(o)
        cv2.rectangle(illu, (o[1].start*2, o[0].start*2), (o[1].stop*2, o[0].stop*2), (random.randint(0,255) ,random.randint(0,255),random.randint(0,255)),1)
        
    for celltype in range(4):
        if celltype == 0: col = (255,0,0)
        if celltype == 1: col = (0,255,0)
        if celltype == 2: col = (255,255,0)
        if celltype == 3: col = (0,0,255)
        for cellid, subline in cells_list[celltype].iteritems():
#             cv2.rectangle(illu, (subline.cellbound[1].start+celltype, subline.cellbound[0].start+celltype), (subline.cellbound[1].stop+celltype, subline.cellbound[0].stop+celltype), col,1)
            line = subline.subline()
            if line is not None:
                pos1 = (int(line[0][0])*2, int(line[0][1])*2)
                pos2 = (int(line[1][0])*2, int(line[1][1])*2)
#                 print cellid, pos1, pos2
                cv2.line(illu, pos1, pos2, col, 1)
    ### illustrate/debug first round
    
    return img_grey, illu
   

class Param(object):
    def __init__(self):
        pass                  

import os
if __name__ == "__main__":
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/tmp/'):        
        if filename[-3:].upper() == 'JPG':
            print filename
            img, illu = extractLines2('/home/loitg/Downloads/complex-bg/tmp/' + filename)
            cv2.imshow('illu', illu)
            cv2.waitKey(-1)
                                   
if __name__ == "__main__2":
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/tmp/'):        
        if filename[-3:].upper() == 'JPG':
            param = Param()
            for k in np.linspace(0.02,0.22,3):
                for w in range(15,56,3):
                    param.k = k
                    param.w = w
                    print filename, k, w
                    binary, img = extractLines('/home/loitg/Downloads/complex-bg/tmp/' + filename, param)
                    img = img.astype(np.uint8)
                    binary = binary.astype(np.uint8)
                    cv2.imshow('line', img[:img.shape[0],:,:])
                    cv2.imshow('bin', binary[:binary.shape[0],:])
                    cv2.waitKey(-1)
#                     suffix = '_'.join([filename, str(k), str(w)])
#                     cv2.imwrite('/home/loitg/Downloads/complex-bg/java/'+suffix+'_img.bmp', img)
#                     cv2.imwrite('/home/loitg/Downloads/complex-bg/java/'+suffix+'_bin.bmp', binary)
            
            
            
            
            
#             
#             
#             while True:
#                 k = extractLines('/home/loitg/Downloads/complex-bg/tmp/' + filename, param)
#                 if k == ord('2'):
#                     param.k += 0.04
#                     print 'k ', param.k
#                 elif k == ord('1'):
#                     param.k -= 0.04
#                     print 'k ', param.k
#                 elif k == ord('4'):
#                     param.w += 7
#                     print 'w ', param.w
#                 elif k == ord('3'):
#                     param.w -= 7
#                     print 'w ', param.w
#                 elif k == ord('n'):
#                     break
                    
# NOTE: w and scale should be near



