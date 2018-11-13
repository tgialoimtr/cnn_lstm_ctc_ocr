#!/usr/bin/env python
import sys
from asn1crypto.x509 import Features
from click.termui import clear
from statsmodels.base.model import Results
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
import math
import hashlib
import colorsys
from statsmodels.nonparametric.smoothers_lowess import lowess
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
from sklearn.externals import joblib

def s(scale, a):
    return int(scale*a)
def v(scale, x,y):
    return (int(scale*x), int(scale*y))

def drawPoints(img, points, col=(255,255,255), scale=1.0):
    for (x,y) in points:
        cv2.circle(img,v(scale, x,y),1, col,-1)
    return img

def drawTBX(img, top, bot, x, col=(255,255,255), scale=1.0):
    cv2.circle(img,v(scale, x,top),1, col,-1)
    cv2.circle(img,v(scale, x,bot), 1, col,-1)
    return img

def str2col(s):
    m = hashlib.md5()
    m.update(s.encode())
    byte_digest = m.digest()
    first_num = float(ord(byte_digest[0]))   # get int value of first character 0-255
    second_num = float(ord(byte_digest[1]))   # get int value of second character 0-255
    hue = first_num / 255  # hue as percentage
    variation = second_num / 255 / 2 - 0.25  # add some limited randomness to saturation and brightness
    saturation = min(0.8 + variation, 1.0)  # will vary from 0.55 to 1.0
    brightness = min(0.7 + variation, 1.0)  # will vary from 0.45 to 0.95
    color = colorsys.hsv_to_rgb(hue, saturation, brightness)
    color = tuple(int(i * 255) for i in color)
    return color


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

class SubLine(object):
    ISTOP = 2
    ISBOT = 1
    ISEMPTY = 0
    IGNORED = -1
    LINE_IOU_COMBINE_THRESHOLD = 0.8
    LINE_SCORE_THRESHOLD = 0.1
    NODE_IN_LINE_THRESHOLD = 0.5
    MIN_INSIDER = 3
    MAX_ABOVE_RATIO = 0.2
    MAX_OUTSIDER_RATIO = 0.9
    
    LOOKAHEAD = 2.5
    INTERVAL_BASIC_LENS = [0.2,0.5]
    MAX_ANGLE = 30.0
    NUM_ANGLE = 31

    def _initCur(self, topy, boty, x):
        self.curtopy = topy
        self.curboty = boty
        self.height = float(self.aboty - self.atopy)
        self.curx = x
        self.curnextx = self.curx + int(self.height * self.LOOKAHEAD * 2)
        ###
        a,b = np.meshgrid(np.arange(self.curx, self.curnextx), self.tanangle_list)
        self.ax_y0 = a*b
        ### 
        self.checkpoints = [(self.curx, self.curtopy - int((self.INTERVAL_BASIC_LENS[0]+self.INTERVAL_BASIC_LENS[1])*self.height)),
                            (self.curx, self.curtopy - int(self.INTERVAL_BASIC_LENS[1]*self.height)),
                            (self.curx, self.curtopy + int(0.5*self.height)),
                            (self.curx, self.curboty + int(self.INTERVAL_BASIC_LENS[1]*self.height)),
                            (self.curx, self.curboty + int((self.INTERVAL_BASIC_LENS[0]+self.INTERVAL_BASIC_LENS[1])*self.height))]

    def __init__(self, topy, boty, x, tops=[], bottoms=[], clf=None, img=None):
        self.tops = tops
        self.bottoms = bottoms
        self.isnew = True
        self.nextCount = 0
        self.id = str(self.curboty) + '.' + str(self.curx)
        self.lastytopindex = 0
        self.lastybotindex = 0
        self.clf = clf
        self.img = img
        # post-set
        self.angle = 0
        self.toppoints = []
        self.botpoints = []
        # others
        self._resetCur(topy, boty, x)
        self.sin = {}; self.cos = {}
        for a in np.linspace(-self.MAX_ANGLE, self.MAX_ANGLE, 9):
            self.sin[a] = math.sin(a/180.0*math.pi)
            self.cos[a] = math.cos(a/180.0*math.pi)
        self.angle_list = np.linspace(-self.MAX_ANGLE, +self.MAX_ANGLE, self.NUM_ANGLE+1)
        self.angle_list = self.angle_list + self.dangle/2
        self.tanangle_list = np.tan(self.angle_list/180.0*math.pi)
        self.dangle = 2.0*self.MAX_ANGLE/self.NUM_ANGLE
        

    def _a2i(self, a):
        index = int(math.floor((a + self.MAX_ANGLE)/self.dangle))
        if index < 0: index = 0
        if index >= self.NUM_ANGLE: index = self.NUM_ANGLE - 1 
        return index
    
    def _i2a(self, index): 
        return self.angle_list[index]
    
    def _i2t(self, index):
        return self.tanangle_list[index]

    def nextRange(self):
        x1 = self.curx
        x2 = self.curnextx
        for x in range(x1 + 1,x2):
            y1 = self.curtopy - int((x-x1)/2.0); 
            y2 = self.curboty + int((x-x1)/2.0);
            for y in range(y1,y2):
                yield x, y
    

    def iou(self, poly1, poly2):
        try:
            poly1 = Polygon(poly1)
            poly2 = Polygon(poly2)
            i = poly1.intersection(poly2).area
            o = poly1.union(poly2).area
        except Exception as e:
            print poly1.exterior.coords.xy
            print poly2.exterior.coords.xy
            print e
            return 1.0
        return 1.0*i/o
    
    def _toAX(self, origin, point):
        delta_x = point[0] - origin[0]
        delta_y = point[1] - origin[1]
        angle = math.atan2(delta_y, delta_x)/math.pi*180.0
        if angle < -self.MAX_ANGLE or angle > +self.MAX_ANGLE:
            return None
        return self._a2i(angle)

    def buildAngleFeatures(self, ax_topA, ax_topB, ax_topC, ax_botB, ax_botC, ax_botD, ax_ytopB, ax_ybotC):                
        ax_topmask = np.cumsum(ax_topB, axis=1).astype(bool)
        ax_botmask = np.cumsum(ax_botC, axis=1).astype(bool)
        ax_mask = ax_topmask & ax_botmask
        widths = np.sum(ax_mask, axis=0)
        countBCfilter = widths > 1
        
        ax_topA = ax_topA[countBCfilter]
        ax_topB = ax_topB[countBCfilter]
        ax_topC = ax_topC[countBCfilter]
        ax_botB = ax_botB[countBCfilter]
        ax_botC = ax_botC[countBCfilter]
        ax_botD = ax_botD[countBCfilter]
        ax_ytopB = ax_ytopB[countBCfilter]
        ax_ybotC = ax_ybotC[countBCfilter]
        ax_mask = ax_mask[countBCfilter]
        
        ax_topA = ax_topA & ax_mask
        ax_topB = ax_topB & ax_mask
        ax_topC = ax_topC & ax_mask
        ax_botB = ax_botB & ax_mask
        ax_botC = ax_botC & ax_mask
        ax_botD = ax_botD & ax_mask

        count_topA = np.sum(ax_topA, axis=0)
        count_topB = np.sum(ax_topB, axis=0)
        count_topC = np.sum(ax_topC, axis=0)
        count_botB = np.sum(ax_botB, axis=0)
        count_botC = np.sum(ax_botC, axis=0)
        count_botD = np.sum(ax_botD, axis=0)

        f0 = np.where(countBCfilter)[0]
        f1 = count_topB
        f2 = count_topA/count_topB
        f3 = count_botB/count_topB
        f4 = count_botC
        f5 = count_botD/count_botC
        f6 = count_topC/count_botC
        
        ff0 = self.angle_list[countBCfilter]
        ff1 = ff0 - self.angle if self.angle is not None else np.zeros_like(ff0)
        ff2 = 1.0*self.height/widths
        ydiff = self.ax_y0 - ax_ytopB
        ydiff[~ax_mask] = 0
        ff3 = np.sum(ydiff**2, axis=0)
        ydiff = self.ax_y0 - ax_ybotC
        ydiff[~ax_mask] = 0
        ff4 = np.sum(ydiff**2, axis=0)
         
        def my_func(row):
            xs = np.where(row)[0].tolist()
            N = len(xs)
            L = max(xs) - min(xs)
            ds = [xs[i+1] - xs[i] for i in range(0,N-1)]
            a = 1.0*L/(N-1)
            b = np.median(ds)
            return abs(a-b)/a
        
        ff5 = np.apply_along_axis(my_func, 0, ax_topB)
        ff6 = np.apply_along_axis(my_func, 0, ax_botC)
        
        
        return np.array([f0,ff0,ff1,f1,f2,f3,f4,f5,f6,ff2,ff3,ff4,ff5,ff6], dtype=float), widths

    def writeFeaturesToFile(self, outfile, islongline, isshortline):
        outfile.write()
        feature

    def buildData(self, features, img, outfile):
#         while True:
#             a = np.random.normal(0.0, self.MAX_ANGLE)
#             i = a2i(a)
#             if i is None: continue
#             features[i] 

        i=0
        while True:
            feature = features[i]
            angle = feature[0]
            width = feature[2]
            
            
            cv2.line(img, (self.curtopy, self.curx), (self.curtopy + int(math.tan(angle/180.0*math.pi)*width), self.curx + int(width)))
            cv2.line(img, (self.curboty, self.curx), (self.curboty + int(math.tan(angle/180.0*math.pi)*width), self.curx + int(width)))
            cv2.imshow('ii', img5)
            k = cv2.waitKey(-1)
            if k == ord('n'):
                return 0
            elif k == ord('1'): #short line
                is_line = True
                is_selected = False
            elif k == ord('2'): #long line
                is_line = True
                is_selected = True
            elif k == ord('`'): # notline
                is_line = False
                is_selected = False
            elif k == ord('a'): # move down
                i += 1
                if i >= len(features): i = len(features) -1
                continue
            elif k == ord('q'): # move up
                i -= 1
                if i < 0: i = 0
                continue
                       
            
            
            
        
        
        
        return selected_results
    
    def _filterCombineConf3(self, results):
        results = results[results[3]]
        if len(results) > 0:
            return results[len(results)/2]
        else:
            return results
    
    def next2(self, allnodes, allpoints, img):        
        ax_topA = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botD = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_ytopB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.float)
        ax_ybotC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.float)
        
        for x,y in self.nextRange():
            if x >= len(allpoints): continue
            if y >= len(allpoints[x]): continue
            point_type = allpoints[x][y]
            if point_type == self.ISEMPTY: continue
            
            alpha_i_s = [self._toAX(chkpoint, (x,y)) for chkpoint in self.checkpoints]
            delta_x = x - self.curx
            delta_x = -delta_x
            
            if point_type == self.ISTOP:
                ax_topA[alpha_i_s[0]:alpha_i_s[1], delta_x] = True
                ax_topB[alpha_i_s[1]:alpha_i_s[2], delta_x] = True
                ax_topC[alpha_i_s[2]:alpha_i_s[3], delta_x] = True
                ax_ytopB[alpha_i_s[1]:alpha_i_s[2], delta_x] = y - (self.checkpoints[1][1] + self.checkpoints[2][1])/2
            elif point_type == self.ISBOT:
                ax_botB[alpha_i_s[1]:alpha_i_s[2], delta_x] = True
                ax_botC[alpha_i_s[2]:alpha_i_s[3], delta_x] = True
                ax_botD[alpha_i_s[3]:alpha_i_s[4], delta_x] = True
                ax_ybotC[alpha_i_s[2]:alpha_i_s[3], delta_x] = y - (self.checkpoints[2][1] + self.checkpoints[3][1])/2
                    
        features, widths = self.buildAngleFeatures(ax_topA, ax_topB, ax_topC, ax_botB, ax_botC, ax_botD, 
                                           ax_ytopB, ax_ybotC)
        
        results = self.buildData(features)

#         results = self.clf.predict(features)
        results = np.concatenate([features[:,0], #id
                                  features[:,1], #angle
                                  widths, #width
                                  (results>0.5)]) #isline
        results = self._filterCombineConf3(results)
        
        retlines = []
        for i, result in enumerate(results):
            idang = result[0]
            angle = result[1]
            width = result[2]

            img5=img.copy()
            
            xmask = ax_topB[idang][:width]
            xs = np.where(xmask)[0]
            ys = ax_ytopB[idang][:width][xmask]
            ys += (self.checkpoints[1][1] + self.checkpoints[2][1])/2
            toppoints = zip(xs, ys)
            drawPoints(img5, toppoints, (0,255,0))
            
            xmask = ax_botC[idang][:width]
            xs = np.where(xmask)[0]
            ys = ax_ybotC[idang][:width][xmask]
            ys += (self.checkpoints[2][1] + self.checkpoints[3][1])/2
            botpoints = zip(xs, ys)
            drawPoints(img5, botpoints, (0,255,0))
            
            topy = self.curtopy + int(math.tan(angle/180.0*math.pi)*width)
            boty = self.curboty + int(math.tan(angle/180.0*math.pi)*width)
            x = self.curx + int(width)
            
            clear
            
            if i == len(results)-1:
                self.curtopy = int(topy)
                self.curboty = int(boty)
                self.curx = int(x)
                self.lastytopindex = len(self.tops)
                self.lastybotindex = len(self.bottoms)
                self.tops = self.tops + toppoints
                self.bottoms = self.bottoms + botpoints
                self.nextCount += 1
                retlines.append(self)
            else:
                another = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=self.tops + toppoints,
                                  bottoms=self.bottoms + botpoints,
                                  clf=self.clf)
                another.lastytopindex = len(self.tops)
                another.lastybotindex = len(self.bottoms)  
                another.isnew = True
                another.nextCount = self.nextCount + 1
                another.id = self.id
                retlines.append(another)

            
        return retlines
    
    
    def smoothedFunc(self, xs, ys):
        _, indices = np.unique(xs, return_index=True) 
        xs = xs[indices]
        ys = ys[indices]
        yhat = lowess(ys, xs, frac=0.666, is_sorted=False, return_sorted=False, delta=0.0)
#         print xs
#         print yhat
        f2 = interp1d(xs, yhat, kind='linear')
        return f2
    
    def extractConstHeight(self, expand=0):
        if len(self.tops) == 0 or len(self.bottoms) == 0:
            return [],[],[]
        tops = np.array(self.tops)
        f_top = self.smoothedFunc(tops[:,0], tops[:,1])
        bottoms = np.array(self.bottoms)
        f_bot = self.smoothedFunc(bottoms[:,0], bottoms[:,1])       
        
        topx1 = min(tops[:,0]); botx1 = min(bottoms[:,0])
        x2 = max(topx1, botx1); x1 = min(topx1, botx1)
        topx2 = max(tops[:,0]); botx2 = max(bottoms[:,0])
        x3 = min(topx2, botx2); x4 = max(topx2, botx2) 
        heights = []
        for x in range(x2,x3):
            heights.append(f_bot(x) - f_top(x))
        height = sum(heights)/len(heights)
        
        f_combined = self.smoothedFunc(np.concatenate([tops[:,0], bottoms[:,0]]), np.concatenate([tops[:,1]+height, bottoms[:,1]]))
        xs = list(range(x2,x3))
        y_bot = [int(f_combined(x)) for x in xs]
        height=int(height)
        y_top = [y - height for y in y_bot]
        return xs, y_top, y_bot

    
    def draw(self, img, col, opacity, drawyhat=True):
        xs, y_top, y_bot = self.extractConstHeight()
        if len(xs) == 0: return
        temp = img.copy()
        for i in range(len(xs)):
            temp[y_top[i]:y_bot[i],xs[i]] = col
        cv2.addWeighted(img, 1-opacity, temp, opacity, gamma=0, dst=img)
        if drawyhat:
            drawPoints(img, zip(xs[::4], y_top[::4]), col)
            drawPoints(img, zip(xs[::4], y_bot[::4]), col)

#         cv2.putText(img,self.id, (xs[0],y_top[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col)
        return img
        
    
    
    def extract(self, img, expand=0):
        xs, y_top, y_bot = self.extractConstHeight()
        n = len(xs)
        if n == 0: return
        height = y_bot[0] - y_top[0]
        if len(img.shape) > 2:
            retline = np.zeros((height,n,3), dtype=np.uint8)
        else:
            retline = np.zeros((height,n), dtype=np.uint8)
        for i in range(n):
            retline[:,i] = img[y_top[i]:y_bot[i],xs[i]]
        return retline  
      
def extractLines2(imgpath):
    img_grey = ocrolib.read_image_gray(imgpath)
    
    (h, w) = img_grey.shape[:2]
    img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
    angle = estimate_skew_angle(img00,linspace(-5,5,42))
    print 'goc', angle
    rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
    
    img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.999, cv2.NORM_MINMAX)

    objects, scale = findBox(img_grey)
    
    ######### convert
    xfrom=0; xto=img_grey.shape[1];
    yfrom=0; yto=min(img_grey.shape[0], 800);
    img_grey = img_grey[yfrom:yto, xfrom:xto]
    objects2 = []
    for obj in objects:
        topy = obj[0].start
        boty = obj[0].stop
        x = (obj[1].start + obj[1].stop)/2
        if yfrom <= topy < yto and yfrom <= boty < yto and xfrom <= x < xto:
            object2 = (slice(obj[0].start - yfrom, obj[0].stop - yfrom, None), slice(obj[1].start - xfrom, obj[1].stop - xfrom, None))
            objects2.append(object2)
            
    objects = objects2
    
    ######### end convert

    h,w = img_grey.shape
    img = (cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
    
    nodes = [[None for j in range(h+1)] for i in range(w+1)]
    points = [[SubLine.ISEMPTY for j in range(h+1)] for i in range(w+1)]
    clearedList = set() ## temporary solution
    
    objects = sorted(objects,key=lambda obj:(obj[1].start + obj[1].stop)/2)
    for bound in objects:
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        top = (x, topy)
        bottom = (x, boty)
        points[bottom[0]][bottom[1]] = SubLine.ISBOT
        points[top[0]][top[1]] = SubLine.ISTOP
        nodes[bottom[0]][bottom[1]] = (topy, boty, x)
            
    allines = []
    
    clf = joblib.load('/home/loitg/Downloads/complex-bg/le_model.pkl')
    
    def move(subline, allnodes, allpoints,img):
        newsublines = subline.next(allnodes, allpoints,img)
        if len(newsublines) > 0:
            for new in newsublines:
                if new.isnew: 
                    allines.append(new)
                    new.isnew = False
                print '______________++++++++++++++=' + new.id
                move(new, allnodes, allpoints, img)
        else:
            subline.clear(allnodes, clearedList)

#     illu = img.copy()
#     for bound in objects:
#         cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].start),2, (255,0,0),-1)
#         cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].stop), 2, (0,255,0),-1)
#         cv2.line(illu, ((bound[1].start + bound[1].stop)/2, bound[0].start), ((bound[1].start + bound[1].stop)/2, bound[0].stop), (0,0,255),1)
#     cv2.imshow('ii', illu)

    for bound in objects: # sorted
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        if (topy, boty, x) in clearedList: 
            continue
        subline = SubLine(topy=topy,
                          boty=boty,
                          x=x,
                          clf=clf)
        allines.append(subline)
        subline.isnew = False
        try:
            move(subline, nodes, points,img)
        except Exception as e:
            pass
    ### illustrate
    img2 = img.copy()
    for line in allines:
        try:
            col = str2col(line.id)
            line.draw(img2, col, 0.5, drawyhat=False)
        except Exception as e:
            pass
#     cv2.imshow('lines', img2)
#     cv2.waitKey(-1)
    
    
    return img2

    
import random, os
if __name__ == "__main__":
    filelist = list(os.listdir('OPIMIZE findBox first !!!!!!'))
    random.shuffle(filelist)
    for filename in filelist:       
#         if filename[-3:].upper() == 'JPG':
        if filename == '12a.JPG':
            print filename
            img = extractLines2('/home/loitg/Downloads/complex-bg/java/' + filename)
#             cv2.imwrite('/home/loitg/Downloads/complex-bg/java4/'+filename, img)
