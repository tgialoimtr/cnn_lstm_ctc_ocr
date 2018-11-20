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
from time import time

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

class LinRay(object):
    
    def __init__(self, b, m):
        self.b = b
        self.m = m
    
    def f(self, x):
        return int(x*self.b + self.m)

def linfit2(xs, ys):
    n = len(xs)
    sumx = np.sum(xs)
    sumy = np.sum(ys)
    sumx2 = np.sum(xs**2)
    sumy2 = np.sum(ys**2)
    sumxy = np.sum(xs*ys)
    denom = (n*sumx2 - sumx*sumx)
    b = 1.0* (n*sumxy - sumx*sumy) / denom
    m = 1.0* (sumy*sumx2 - sumx*sumxy) /denom
    s2e = (n*sumy2 - sumy*sumy - b*b*denom)/(n*(n-2))
    
    return b,m,s2e

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
    LINE_SCORE_THRESHOLD = 0.1
    NODE_IN_LINE_THRESHOLD = 0.5
    
    LOOKAHEAD = 5.0
    INTERVAL_BASIC_LENS = [0.4,0.2]
    MAX_ANGLE = 15.0
    NUM_ANGLE = 10
    CB_LEN = 5.0
    SCLINE_LEFT = 5.0
    SCLINE_RIGHT = 5.0
    EP_LEN = 2.0
    EP_ABOVE = 0.5
    EP_BELOW = 0.5
    EP_FINAL_EXPAND = (0.2, 1.0)
    

    def _initCur(self):
        self.height = float(self.curboty - self.curtopy)
        self.curnextx = self.curx + int(self.height * self.LOOKAHEAD)
        if self.curnextx > self.imgwidth: self.curnextx = self.imgwidth
        ###
        a,b = np.meshgrid(np.arange(self.curnextx-self.curx, 0,-1), self.tanangle_list)
        self.ax_y0 = a*b
        ### 
        self.checkpoints = [(self.curx, self.curtopy - int((self.INTERVAL_BASIC_LENS[0]+self.INTERVAL_BASIC_LENS[1])*self.height)),
                            (self.curx, self.curtopy - int(self.INTERVAL_BASIC_LENS[1]*self.height)),
                            (self.curx, self.curtopy + int(0.5*self.height)),
                            (self.curx, self.curboty + int(self.INTERVAL_BASIC_LENS[1]*self.height)),
                            (self.curx, self.curboty + int((self.INTERVAL_BASIC_LENS[0]+self.INTERVAL_BASIC_LENS[1])*self.height))]

    def __init__(self, topy, boty, x, tops=[], bottoms=[], clf=None, img=None, angle=None, pointsmap=None):
        self.id = str(boty) + '.' + str(x)
        self.curtopy = topy
        self.curboty = boty
        self.curx = x
        
        self.pointsmap = pointsmap
        self.img = img
        self.imgheight = img.shape[0]
        self.imgwidth = img.shape[1]
        self.tops = tops
        self.bottoms = bottoms
        self.isnew = True
        self.nextCount = 0

        self.clf = clf
        # post-set
        self.angle = angle
        # others
        self.angle_list = np.linspace(-self.MAX_ANGLE, +self.MAX_ANGLE, self.NUM_ANGLE+1)
        self.angle_list = self.angle_list[:-1]
        self.dangle = 2.0*self.MAX_ANGLE/self.NUM_ANGLE
        self.angle_list = self.angle_list + self.dangle/2
        self.tanangle_list = np.tan(self.angle_list/180.0*math.pi)
        self._initCur() 
        # combine line
        self._initCombineInfo()


    def _a2i(self, a):
        index = int(math.floor((a + self.MAX_ANGLE)/self.dangle))
        return index
    
    def _i2a(self, index): 
        return self.angle_list[index]
    
    def _i2t(self, index):
        return self.tanangle_list[index]
    
    def _t2a(self, t):
        return math.atan(t)/math.pi*180.0

    def nextPoints(self):
        x1 = self.curx + 1
        x2 = self.curnextx
        maxdeltay = int(self.ax_y0[-1,0] + 1)
        y1 = self.curtopy - maxdeltay; y1 = self._clipy(y1)
        y2 = self.curboty + maxdeltay
        subwindow = self.pointsmap[y1:y2, x1:x2]
        toppoints = np.where(subwindow==SubLine.ISTOP)
        botpoints = np.where(subwindow==SubLine.ISBOT)
        return zip(toppoints[1]+x1, toppoints[0]+y1), zip(botpoints[1]+x1, botpoints[0]+y1)
    

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
        return self._a2i(angle)

    def buildAngleFeatures(self, ax_topA, ax_topB, ax_topC, ax_botB, ax_botC, ax_botD, ax_ytopB, ax_ybotC):        
        ax_topmask = np.cumsum(ax_topB, axis=1).astype(bool)
        ax_botmask = np.cumsum(ax_botC, axis=1).astype(bool)
        ax_mask = ax_topmask & ax_botmask
#         print 'topB 0'
#         print ax_topB.astype(uint8)
#         print 'botB 0'
#         print ax_botB.astype(uint8)
#         print 'botC 0'
#         print ax_botC.astype(uint8)
#         print 'topC 0'
#         print ax_topC.astype(uint8)
#         print 'mask 0'
#         print ax_mask.astype(uint8)
        ax_topB = ax_topB & ax_mask
        ax_botC = ax_botC & ax_mask
        count_topB = np.sum(ax_topB, axis=1)
        count_botC = np.sum(ax_botC, axis=1)
        countBCfilter = (count_topB > 1) & (count_botC > 1)
#         print 'filter'
#         print countBCfilter.astype(uint8)
        ax_mask = ax_mask[countBCfilter] 
#         print 'mask 1'
#         print ax_mask.astype(uint8)
        widths = np.sum(ax_mask, axis=1)
#         print 'width'
#         print widths.astype(uint8)
        if np.all(~countBCfilter):
            return np.zeros(shape=(0,14),dtype=float), widths
        
#         print 'ax_y0'
#         print self.ax_y0
#         print 'ytopB'
#         print ax_ytopB
        ax_topA = ax_topA[countBCfilter]
        ax_topB = ax_topB[countBCfilter]
        ax_topC = ax_topC[countBCfilter]
        ax_botB = ax_botB[countBCfilter]
        ax_botC = ax_botC[countBCfilter]
        ax_botD = ax_botD[countBCfilter]
        ax_ytopB = ax_ytopB[countBCfilter]
        ax_ybotC = ax_ybotC[countBCfilter]
        ax_y0 = self.ax_y0[countBCfilter]

#         print 'topB 1'
#         print ax_topB.astype(uint8)
#         print 'botC 1'
#         print ax_botC.astype(uint8)
        
        ax_topA = ax_topA & ax_mask
        ax_topC = ax_topC & ax_mask
        ax_botB = ax_botB & ax_mask
        ax_botD = ax_botD & ax_mask

        count_topA = np.sum(ax_topA, axis=1)
        count_topB = np.sum(ax_topB, axis=1)
        count_topC = np.sum(ax_topC, axis=1)
        count_botB = np.sum(ax_botB, axis=1)
        count_botC = np.sum(ax_botC, axis=1)
        count_botD = np.sum(ax_botD, axis=1)
#         print 'count topA'
#         print count_topA
#         print 'count topB'
#         print count_topB
#         print 'count topC'
#         print count_topC
#         print 'count botB'
#         print count_botB
#         print 'count botC'
#         print count_botC  
#         print 'count botD'
#         print count_botD   
        f0 = np.where(countBCfilter)[0]
        f1 = 1.0*count_topB
        f2 = 1.0*count_topA/count_topB
        f3 = 1.0*count_botB/count_topB
        f4 = 1.0*count_botC
        f5 = 1.0*count_botD/count_botC
        f6 = 1.0*count_topC/count_botC
        
        ff0 = self.angle_list[countBCfilter]
        ff1 = ff0 - self.angle if self.angle is not None else np.zeros_like(ff0)
        ff2 = 1.0*self.height/widths

        ydiff = ax_y0 - ax_ytopB
        ydiff[~ax_mask] = 0
        ff3 = np.sqrt(np.sum(ydiff**2, axis=1))/self.height/count_topB
        ydiff = ax_y0 - ax_ybotC
        ydiff[~ax_mask] = 0
        ff4 = np.sqrt(np.sum(ydiff**2, axis=1))/self.height/count_botC
        
        def my_func(row):
#             print 'row ' + str(row.astype(uint8))
            xs = np.where(row)[0].tolist() + [len(row)]
#             print 'xs ' + str(xs)
            N = len(xs)
            L = xs[-1] - xs[0]
            ds = [xs[i+1] - xs[i] for i in range(0,N-1)]
            a = 1.0*L/(N-1)
            b = np.median(ds)
#             print a,b, abs(a-b)/a
            return abs(a-b)/a

        ff5 = np.apply_along_axis(my_func, 1, ax_topB)
        ff6 = np.apply_along_axis(my_func, 1, ax_botC)

        return np.transpose(np.array([f0,ff0,ff1,f1,f2,f3,f4,f5,f6,ff2,ff3,ff4,ff5,ff6], dtype=float)), widths

    def writeFeaturesToFile(self, features, islongline, isshortline):
        features = features.tolist() + [islongline, isshortline]
        features = [str(f) for f in features]
        with open('/home/loitg/Downloads/complex-bg/le_cls_data_3.csv','a') as outfile:
            outfile.write(','.join(features) + '\n')

    def buildData_dummy(self, features, widths):
        n = len(features)
        if n == 0: return np.array([])
        temp = (features[:,1] < 5) &  (features[:,1] > -5)
        return temp.astype(float)
    
    def buildData(self, features, widths):
        i=0
        n = len(features)
        if n == 0: return np.array([])
        results = np.zeros(n)
        while True:
            feature = features[i]
            angle = feature[1]
            width = widths[i]
            fea = zip(range(len(feature)), feature)
            fstr = [str(x) for x in fea]
            print(''.join(fstr) )
            img5 = self.img.copy()
            cv2.line(img5, (self.curx, self.curtopy), (self.curx + int(width), self.curtopy + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),1)
            cv2.line(img5, (self.curx, self.curboty), (self.curx + int(width), self.curboty + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),1)
            cv2.imshow('ii', img5)
            k = cv2.waitKey(-1)
            if k == ord('n'):
                break 
            elif k == ord('1'): #short line
                self.writeFeaturesToFile(feature[1:], 0, 1)
                results[i] = 1
            elif k == ord('2'): #long line
                self.writeFeaturesToFile(feature[1:], 1, 0)
                results[i] = 1
            elif k == ord('`'): # notline
                self.writeFeaturesToFile(feature[1:], 0, 0)
            elif k == ord('j'): # move down
                i += 1
                if i >= n: i = n -1
                continue
            elif k == ord('i'): # move up
                i -= 1
                if i < 0: i = 0
                continue
            elif k == ord('k'): # move down
                i += 5
                if i >= n: i = n -1
                continue
            elif k == ord('o'): # move up
                i -= 5
                if i < 0: i = 0
                continue
        
        return np.array(results)
    
    def _filterCombineConf3(self, results):
#         results = results[results[:,3]>0.5]
#         ids = results
#         angle = result[1]
#         width = int(result[2])

        
        if len(results) > 0:
            ret_filter = np.zeros((len(results),), dtype=bool)
            ret_indices = []
            a = (results[:,4]>0.5).astype(int8)
            c = np.where(np.diff(a)!=0)[0] + 1
            d = [0] + c.tolist() + [len(a)]
            for i in range(0,len(d)-1):
                cont_res = results[d[i]:d[i+1],:]
                max_width_index = np.argmax(cont_res[:,2])
                if cont_res[0,4] > 0.5:
                    ret_indices.append((len(cont_res), int(2*cont_res[max_width_index,2]/self.height), -cont_res[max_width_index,3], max_width_index + d[i]))
            
            ret_indices.sort(reverse=True)
            if len(ret_indices) > 0:
                ret_filter[ret_indices[0][3]] = True
            return results[ret_filter]
            
        else:
            return results
    
    def next2(self): 
        retlines = []
               
        ax_topA = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botD = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_ytopB = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.float)
        ax_ybotC = np.zeros((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.float)
        
        toppoints, botpoints = self.nextPoints()
        
        def clipaa(a):
            if a < 0: a=0
            if a >= self.NUM_ANGLE: a = self.NUM_ANGLE - 1
            return a
            
        for x,y in toppoints:
            alpha_i_s = [self._toAX(chkpoint, (x,y)) for chkpoint in self.checkpoints]
            alpha_i_s = [clipaa(a) for a in alpha_i_s]
            delta_x = x - self.curx
            delta_x = -delta_x
            ax_topA[alpha_i_s[1]:alpha_i_s[0], delta_x] = True
            ax_topB[alpha_i_s[2]:alpha_i_s[1], delta_x] = True
            ax_topC[alpha_i_s[3]:alpha_i_s[2], delta_x] = True
            ax_ytopB[alpha_i_s[2]:alpha_i_s[1], delta_x] = y - (self.checkpoints[1][1] + self.checkpoints[2][1])/2
                
        for x,y in botpoints:
            alpha_i_s = [self._toAX(chkpoint, (x,y)) for chkpoint in self.checkpoints]
            alpha_i_s = [clipaa(a) for a in alpha_i_s]
            delta_x = x - self.curx
            delta_x = -delta_x
            ax_botB[alpha_i_s[2]:alpha_i_s[1], delta_x] = True
            ax_botC[alpha_i_s[3]:alpha_i_s[2], delta_x] = True
            ax_botD[alpha_i_s[4]:alpha_i_s[3], delta_x] = True
            ax_ybotC[alpha_i_s[3]:alpha_i_s[2], delta_x] = y - (self.checkpoints[2][1] + self.checkpoints[3][1])/2
        
        features, widths = self.buildAngleFeatures(ax_topA, ax_topB, ax_topC, ax_botB, ax_botC, ax_botD, 
                                           ax_ytopB, ax_ybotC)
#         results = self.buildData(features, widths)
        
        if len(features) == 0: 
            results = np.array([])
        else:
            results = self.clf.predict(features[:,1:])
#         positives = np.where(results > 0.5)[0]
#         img5 = self.img.copy()
#         cv2.circle(img5,(self.curx, self.curtopy),5, (255,0,0),-1)
#         cv2.circle(img5,(self.curx, self.curboty), 5, (0,255,0),-1)
#         for i in positives:
#             feature = features[i]
#             angle = feature[1]
#             width = widths[i]
#                   
#             cv2.line(img5, (self.curx, self.curtopy), (self.curx + int(width), self.curtopy + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),1)
#             cv2.line(img5, (self.curx, self.curboty), (self.curx + int(width), self.curboty + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),1)
#         cv2.imshow('ii', img5)
#         k = cv2.waitKey(-1)
        
        
        results = np.array([features[:,0]+0.1, #id
                                  features[:,1], #angle
                                  widths + 0.1, #width
                                  np.abs(features[:,2]), #angle diff
                                  results]) #isline
        results = np.transpose(results)
        results = self._filterCombineConf3(results)
#         for res in results:
#             angle = res[1]
#             width = int(res[2])
#                   
#             cv2.line(img5, (self.curx, self.curtopy), (self.curx + int(width), self.curtopy + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),2)
#             cv2.line(img5, (self.curx, self.curboty), (self.curx + int(width), self.curboty + int(math.tan(angle/180.0*math.pi)*width)),  (0,0,255),2)
#         cv2.imshow('ii', img5)
#         k = cv2.waitKey(-1)
        
        for i, result in enumerate(results):
            idang = int(result[0])
            angle = result[1]
            width = int(result[2])

            img5=self.img.copy()


            xmask = ax_botC[idang][:-(width+1):-1]
            xs = np.where(xmask)[0]
            ys = ax_ybotC[idang][:-(width+1):-1][xmask].astype(int)
            ys += (self.checkpoints[2][1] + self.checkpoints[3][1])/2
            xs += self.curx + 1
            botpoints = [(self.curx, self.curboty)] + zip(xs, ys)
            boty = (self.curboty + int(self._i2t(idang)*width) + np.max(ys[-min(len(ys),3)/2:]))/2
            drawPoints(img5, botpoints, (0,0,120))
            
            
            xmask = ax_topB[idang][:-(width+1):-1]
            xs = np.where(xmask)[0]
            ys = ax_ytopB[idang][:-(width+1):-1][xmask].astype(int)
            ys += (self.checkpoints[1][1] + self.checkpoints[2][1])/2
            xs += self.curx + 1
            toppoints = [(self.curx, self.curtopy)] + zip(xs, ys) 
            topy = (self.curtopy + int(self._i2t(idang)*width) + np.max(ys[-min(len(ys),3)/2:]))/2
            drawPoints(img5, toppoints, (0,0,255))
            
            x = self.curx + int(width)

#             cv2.imshow('gg', img5)
#             cv2.waitKey(-1)
            
            if i == len(results)-1:
                self.curtopy = int(topy)
                self.curboty = int(boty)
                self.curx = int(x)
                self.tops = self.tops + toppoints
                self.bottoms = self.bottoms + botpoints
                self.angle = angle
                self.nextCount += 1
                self._initCur()
                retlines.append(self)
            else:
                raise ValueError
                another = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=self.tops + toppoints,
                                  bottoms=self.bottoms + botpoints,
                                  clf=self.clf,
                                  img=self.img,
                                  pointsmap=self.pointsmap,
                                  angle = angle) 
                another.isnew = True
                another.nextCount = self.nextCount + 1
                another.id = self.id
                retlines.append(another)

            
        return retlines
    
    

    
    def _initCombineInfo(self):
        self.available = True
        self.emptyGaps = []
        self.topright = None
        self.topleft = None
        self.botright = None
        self.botleft = None
        self.bounds = None
        self.leftray = None
        self.rightray = None

    def _updateCombineInfo(self):
        self.tops.sort()
        self.bottoms.sort()
        
#         img3 = self.img.copy()
#         drawPoints(img3, self.bottoms, (0,0,255), 1)
                   
        bottoms = np.array(self.bottoms)
        x4,y4 = bottoms[-1]
        x1,y1 = bottoms[0]
        cb_len = int(self.height*self.CB_LEN)
        if x4 - x1 > self.height*self.CB_LEN*1.5:
            x2 = x1 + cb_len
            x3 = x4 - cb_len
            left_filter = (bottoms[:,0] >= x1) & (bottoms[:,0] <= x2)
            if np.sum(left_filter) <=3:
                lefts = bottoms
            else:
                lefts = bottoms[left_filter,:]
#             drawPoints(img3, zip(lefts[:,0], lefts[:,1]), (255,0,0), 1)
            bl,ml,_ = linfit2(lefts[:,0], lefts[:,1])
            self.leftray = LinRay(bl, ml)
            
            right_filter = (bottoms[:,0] >= x3) & (bottoms[:,0] <= x4)
            if np.sum(right_filter) <=3:
                rights = bottoms
            else:
                rights = bottoms[right_filter,:]
#             drawPoints(img3, zip(rights[:,0], rights[:,1]), (0,255,0), 1)
            br,mr,_ = linfit2(rights[:,0], rights[:,1])
            self.rightray = LinRay(br, mr)
        else:
            bc,mc,e = linfit2(bottoms[:,0], bottoms[:,1])
            self.leftray = LinRay(bc, mc)
            self.rightray = self.leftray
        ### 
        self.topright = self.tops[-1]
        self.topleft = self.tops[0]
        self.botright = (x4,y4)
        self.botleft = (x1,y1)
        ###
        maxy = max(bottoms[:,1])
        miny = min([point[1] for point in self.tops])
        minx = min(x1, self.topleft[0])
        maxx = max(x4, self.topright[0])
        self.bounds = (slice(miny, maxy, None), slice(minx, maxx, None))
        
        
        
#         cv2.line(img3, (0, int(self.rightray.m)), (self.imgwidth, \
#             int(self.rightray.b*self.imgwidth + self.rightray.m)), (0,255,0),1)
#         cv2.line(img3, (0, int(self.leftray.m)), (self.imgwidth, \
#             int(self.leftray.b*self.imgwidth + self.leftray.m)), (255,0,0),1)
#         cv2.imshow('ray', img3)
#         cv2.waitKey(-1)


    def expandPoints(self):
        xs, y_top, y_bot = self.curveline
        height = y_bot[0] - y_top[0]
        ep_len = int(height*self.EP_LEN)
        ep_above = int(height*self.EP_ABOVE)
        ep_below = int(height*self.EP_BELOW)
        ep_middle = int(height*0.5)
        x1 = self._clipx(xs[0] - ep_len)
        x4 = self._clipx(xs[-1] + ep_len)
        y1 = self._clipy(self.bounds[0].start - ep_above)
        y4 = self._clipy(self.bounds[0].stop + ep_below)
        subwindow = self.pointsmap[y1:y4, x1:x4]
        toppoints = np.where(subwindow==SubLine.ISTOP)
        botpoints = np.where(subwindow==SubLine.ISBOT)
        origin = [y1, x1]
        toppoints = np.array(toppoints).T + origin
        botpoints = np.array(botpoints).T + origin
        def expandThisTopPoint(point):
            y,x = point
            if x < xs[0]:
                y0 = self.leftray.f(x) - height
            elif x <= xs[-1]:
                y0 = y_top[x-xs[0]]
            elif x <= x4:
                y0 = self.rightray.f(x) - height
            if y > y0 - ep_above and y < y0 + ep_middle:
                return True
            else:
                return False
        def expandThisBotPoint(point):
            y,x = point
            if x < xs[0]:
                y0 = self.leftray.f(x)
            elif x <= xs[-1]:
                y0 = y_bot[x-xs[0]]
            elif x <= x4:
                y0 = self.rightray.f(x)
            if y > y0 - ep_middle and y < y0 + ep_below:
                return True
            else:
                return False
            
        img4 = self.img.copy()
#         drawPoints(img4, self.tops, (255,0,0))
        validtop_filter = np.apply_along_axis(expandThisTopPoint, 1, toppoints)
        validtops = toppoints[validtop_filter]
        self.tops = [(point[1], point[0]) for point in validtops]
#         drawPoints(img4,[(point[1], point[0]) for point in validtops], (0,0,125))
        self.tops.sort()

        drawPoints(img4, self.bottoms, (0,255,0))
        validbot_filter = np.apply_along_axis(expandThisBotPoint, 1, botpoints)
        validbots = botpoints[validbot_filter]
        self.bottoms = [(point[1], point[0]) for point in validbots]
#         drawPoints(img4,[(point[1], point[0]) for point in validbots], (0,0,255))
        self.bottoms.sort()
        
#         cv2.imshow('ep', img4)
#         cv2.waitKey(-1)

    def scoreLineAfter(self, line_after):
#         img3 = self.img.copy()
#         line_after.draw(img3, str2col(line_after.id), 0.5, drawyhat=False)
#         cv2.rectangle(img3, (line_after.bounds[1].start, line_after.bounds[0].start), 
#                       (line_after.bounds[1].stop, line_after.bounds[0].stop), 1, 2)
#         print 'their bound ', str(line_after.bounds)
#         cv2.rectangle(img3, (self.bounds[1].start, self.bounds[0].start), 
#                       (self.bounds[1].stop, self.bounds[0].stop), 1, 2)
#         cv2.line(img3, (0, int(self.rightray.m)), (self.imgwidth, \
#             int(self.rightray.b*self.imgwidth + self.rightray.m)), str2col(self.id),1)
#         cv2.line(img3, (0, int(line_after.rightray.m)), (line_after.imgwidth, \
#             int(line_after.rightray.b*line_after.imgwidth + line_after.rightray.m)), (255,0,0),1)
#         print 'this bound ', str(self.bounds)
#         cv2.line(img3, (0, int(line_after.leftray.m)), (line_after.imgwidth, \
#             int(line_after.leftray.b*line_after.imgwidth + line_after.leftray.m)), (0,255,0),1)
#         cv2.imshow('current-line', img3)
        
#         if line_after.bounds[0].start > self.bounds[0].stop or line_after.bounds[0].stop < self.bounds[0].start: 
#             print 'outbound'
#             return False, -1
#         if line_after.bounds[1].start > self.bounds[1].stop or line_after.bounds[1].stop < self.bounds[1].start: 
#             print 'outbound'
#             return False, -1
        
        x_this, y_this = self.botright 
        x_after, y_after = line_after.botleft
        y_after_expected = self.rightray.f(x_after)
        maxheight = max(self.height, line_after.height)
        y1 = y_after_expected - int(0.5*maxheight)
        y2 = y_after_expected + int(0.5*maxheight)
        x1 = x_this - int(maxheight*self.SCLINE_LEFT)
        x2 = x_this + int(maxheight*self.SCLINE_RIGHT)
#         print x_this, x_after, y_after, y_after_expected, maxheight
#         print y1, y2, x1, x2
        if y_after < y2 and y_after > y1:
            if x_after <= self.botleft[0]:
#                 print 'outx'
                return False, -1
            elif x_after < x1:
                delta_a = abs(self.rightray.b - line_after.rightray.b)
                if delta_a <= 0.055: # arctan(3 degree)
#                     print 'accept-left'
#                     print self._t2a(self.rightray.b), self._t2a(line_after.rightray.b)
                    return True, 0
                else:
#                     print 'out-leftangle'
#                     print self._t2a(self.rightray.b), self._t2a(line_after.rightray.b)
                    return False, -1
            elif x_after > x2:
                delta_a = abs(self.rightray.b- line_after.leftray.b)
                width_ratio_after = 1.0*(self.bounds[1].stop - self.bounds[1].start)/maxheight
                if delta_a <=0.055 and width_ratio_after > 1.8 : # arctan(3 degree) 
                    delta_x = x_after - x_this
#                     print 'accept-right'
#                     print self._t2a(self.rightray.b), self._t2a(line_after.leftray.b), width_ratio_after
                    return False, delta_x
                else:
#                     print 'out-rightangle-width'
#                     print self._t2a(self.rightray.b), self._t2a(line_after.leftray.b), width_ratio_after
                    return False, -1
            else:
#                 print 'accept-center'
                return True, 0
        else:
#             print 'outy'
            return False, -1
        
#         use combineinfo
#         return instantAdd, score
    
    def combineLinesAfter(self, lines_after):
        for line in lines_after:
            self.tops += line.tops
            self.bottoms += line.bottoms
            self.nextCount += line.nextCount
            line.available = False
# TODO:            if line far: set gaps from bounds
        self._updateCombineInfo()      

    def _clipx(self, a):
        if a < 0: return 0
        if a >= self.imgwidth: return self.imgwidth - 1
        return a

    def _clipy(self, a):
        if a < 0: return 0
        if a >= self.imgheight: return self.imgheight - 1
        return a
      
    def clear(self, cleared_maps):
        xs, y_top, y_bot = self.curveline
        n = len(xs)
        if n == 0: return
        height = y_bot[0] - y_top[0]
        threshold = int(height*SubLine.NODE_IN_LINE_THRESHOLD)
        for i in range(n):
            cleared_maps[0][self._clipy(y_top[i] - threshold):self._clipy(y_top[i] + threshold),xs[i]] = True
            cleared_maps[1][self._clipy(y_bot[i] - threshold):self._clipy(y_bot[i] + threshold),xs[i]] = True
    
    def smoothedFunc(self, xs, ys):
        _, indices = np.unique(xs, return_index=True) 
        xs = xs[indices]
        ys = ys[indices]
        if self.nextCount > 1:
            yhat = lowess(ys, xs, frac=0.666, is_sorted=False, return_sorted=False, delta=0.0)
        else:
            b,m,_= linfit2(xs, ys)
            yhat = (xs*b + m).astype(int)
            
#         from matplotlib import pyplot as plt
#         plt.plot(xs, ys, 'bs', xs, yhat, 'g^')
#         plt.show()
        f2 = interp1d(xs, yhat, kind='linear')
        return f2
    
    # reset tops and bottoms sould call this
    def _updateCurve(self):
        if len(self.tops) == 0 or len(self.bottoms) == 0:
            self.curveline = [],[],[]
        else:
            tops = np.array(self.tops)
            f_top = self.smoothedFunc(tops[:,0], tops[:,1])
            bottoms = np.array(self.bottoms)
            f_bot = self.smoothedFunc(bottoms[:,0], bottoms[:,1])       
            
            topx1 = min(tops[:,0]); botx1 = min(bottoms[:,0])
            x2 = max(topx1, botx1); #x1 = min(topx1, botx1)
            topx2 = max(tops[:,0]); botx2 = max(bottoms[:,0])
            x3 = min(topx2, botx2); #x4 = max(topx2, botx2) 
            heights = []
            for x in range(x2,x3):
                heights.append(f_bot(x) - f_top(x))
            height = sum(heights)/len(heights)
            
            xs = list(range(x2,x3))
            y_bot = [int(f_bot(x)) for x in xs]
            height=int(height)
            y_top = [y - height for y in y_bot]
            self.curveline = np.array(xs), np.array(y_top), np.array(y_bot)
    
#         f_combined = self.smoothedFunc(np.concatenate([tops[:,0], bottoms[:,0]]), np.concatenate([tops[:,1]+height, bottoms[:,1]]))
#         xs = list(range(x2,x3))
#         y_bot = [int(f_combined(x)) for x in xs]
#         height=int(height)
#         y_top = [y - height for y in y_bot]
#         return xs, y_top, y_bot

    ### Must call in or after _createCurve()
    def draw(self, img, col, opacity, drawyhat=True, drawline=False):
        xs, y_top, y_bot = self.curveline
        if len(xs) == 0: return
        temp = img.copy()
        for i in range(len(xs)):
            temp[y_top[i]:y_bot[i],xs[i]] = col
        if drawline:
            b,m,_= linfit2(xs, y_bot)
            cv2.line(temp, (0, int(m)), (self.imgwidth, int(b*self.imgwidth + m)),  col,1)
        cv2.addWeighted(img, 1-opacity, temp, opacity, gamma=0, dst=img)
        if drawyhat:
            drawPoints(img, zip(xs[::4], y_top[::4]), col)
            drawPoints(img, zip(xs[::4], y_bot[::4]), col)

#         cv2.putText(img,self.id, (xs[0],y_top[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col)
        return img
        
    ### Must call in or after _createCurve()
    def extract(self, img, expands=(0.0,0.0)):
        xs, y_top, y_bot = self.curveline
        n = len(xs)
        if n == 0: return
        height = y_bot[0] - y_top[0]
        ep_abovebelow = int(height*expands[0])
        ep_leftright = int(height*expands[1])
        
        x1 = self._clipx(xs[0] - ep_leftright)
        x1s = np.arange(x1, xs[0])
        y1_bot = np.array([self.leftray.f(x) for x in x1s])
        y1_bot_ep = y1_bot + ep_abovebelow
        y1_top_ep = y1_bot - height - ep_abovebelow
        
        x4 = self._clipx(xs[-1] + ep_leftright)
        x4s = np.arange(xs[-1]+1, x4)
        y4_bot = np.array([self.rightray.f(x) for x in x4s])
        y4_bot_ep = y4_bot + ep_abovebelow
        y4_top_ep = y4_bot - height - ep_abovebelow
        
        xs_ep = np.concatenate([x1s, xs, x4s])
        y_top_ep = np.concatenate([y1_top_ep, y_top- ep_abovebelow, y4_top_ep])
        y_bot_ep = np.concatenate([y1_bot_ep, y_bot+ ep_abovebelow, y4_bot_ep])
        y_top_ep = np.clip(y_top_ep,0,self.imgheight-1)
        y_bot_ep = np.clip(y_bot_ep,0,self.imgheight-1)
        n = len(xs_ep)
        height = y_bot_ep[0] - y_top_ep[0]
        if len(self.img.shape) > 2:
            retline = np.zeros((height,n,3), dtype=np.uint8)
        else:
            retline = np.zeros((height,n), dtype=np.uint8)
        for i in range(n): #TODO: remove EmptyGaps
            aa = y_top_ep[i]
            bb = y_bot_ep[i]
            retline[:(bb-aa),i] = self.img[aa:bb,xs_ep[i]]
        return retline 


def extractLines2(imgpath):
    clf = joblib.load('/home/loitg/Downloads/complex-bg/le_model_3.pkl')
    tt = time()
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
#     xfrom=50; xto=img_grey.shape[1];
#     yfrom=700; yto=1500#min(img_grey.shape[0], 800);
#     img_grey = img_grey[yfrom:yto, xfrom:xto]
#     objects2 = []
#     for obj in objects:
#         topy = obj[0].start
#         boty = obj[0].stop
#         x = (obj[1].start + obj[1].stop)/2
#         if yfrom <= topy < yto and yfrom <= boty < yto and xfrom <= x < xto:
#             object2 = (slice(obj[0].start - yfrom, obj[0].stop - yfrom, None), slice(obj[1].start - xfrom, obj[1].stop - xfrom, None))
#             objects2.append(object2)
#              
#     objects = objects2
    
    ######### end convert

    h,w = img_grey.shape
    img = (cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)

    cleared_maps = np.zeros((2,h,w), dtype=bool)
    pointsmap = np.zeros((h,w), dtype=np.uint8)
    
    objects = sorted(objects,key=lambda obj:(obj[1].start + obj[1].stop)/2)
    for bound in objects:
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        if topy >= h or boty >= h or x >= w: continue
        pointsmap[boty,x] = SubLine.ISBOT
        pointsmap[topy,x] = SubLine.ISTOP
            
    allines = []

    illu = img.copy()
    for bound in objects:
        cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].start),2, (255,0,0),-1)
        cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].stop), 2, (0,255,0),-1)
 
#         cv2.line(illu, ((bound[1].start + bound[1].stop)/2, bound[0].start), ((bound[1].start + bound[1].stop)/2, bound[0].stop), (0,0,255),1)  
    
    
    def move(subline):
        newsublines = subline.next2()
        if len(newsublines) > 0:
            for new in newsublines:
                move(new)
        elif subline.nextCount > 0:
            subline._updateCurve()
            subline.clear(cleared_maps)
            subline._updateCombineInfo()
            allines.append(subline)
#             if self.nextCount > 1:
#                 temp1 = cv2.addWeighted(cv2.cvtColor((cleared_maps[0]*120).astype(uint8),cv2.COLOR_GRAY2BGR), 0.5, illu, 0.5,0)
#                 temp2 = cv2.addWeighted(cv2.cvtColor((cleared_maps[1]*120).astype(uint8),cv2.COLOR_GRAY2BGR), 0.5, illu, 0.5,0)
#                 cv2.imshow('bb', cv2.addWeighted(temp1,0.5,temp2,0.5,0))
#                 cv2.waitKey(-1)


    for bound in objects: # sorted
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        if boty - topy < 12: continue
        try:
            if cleared_maps[0][topy,x] and cleared_maps[1][boty,x]:
                continue
        except Exception as e:
            continue
        subline = SubLine(topy=topy,
                          boty=boty,
                          x=x,
                          clf=clf,
                          img=img,
                          pointsmap=pointsmap)
        move(subline)
    
    allines.sort(key=lambda x: x.bounds[1].stop)
    i = 0
    while i < len(allines):
        result = allines[i]
        if result.available:
#             print '-------------------------'
#             img2 = illu.copy()
#             result.draw(img2, (125,125,125), 0.5, drawyhat=False)
#             cv2.line(img2, (0, int(result.rightray.m)), (result.imgwidth, \
#                 int(result.rightray.b*result.imgwidth + result.rightray.m)), (255,0,0),1)
#             cv2.line(img2, (0, int(result.leftray.m)), (result.imgwidth, \
#                 int(result.leftray.b*result.imgwidth + result.leftray.m)), (0,255,0),1)
#             cv2.imshow('cb', img2)
#             cv2.waitKey(-1)
            
            linemap = []
            forceCombines = []
            for j in range(i, len(allines)):
                if j==i: continue
                candidate = allines[j]
                if not candidate.available: continue
                forcedCombined, score = result.scoreLineAfter(candidate)
#                 cv2.waitKey(-1) ####################
                if forcedCombined:
                    forceCombines.append(candidate)
                elif score >= 0:
                    linemap.append((score, candidate))
            result.combineLinesAfter(forceCombines)
            
            
#             for fcb in forceCombines:
#                 col = str2col(fcb.id)
#                 fcb.draw(img2, col, 0.5, drawyhat=False)
#                 cv2.line(img2, (0, int(fcb.rightray.m)), (fcb.imgwidth, \
#                     int(fcb.rightray.b*fcb.imgwidth + fcb.rightray.m)), (255,0,0),1)
#                 cv2.line(img2, (0, int(fcb.leftray.m)), (fcb.imgwidth, \
#                     int(fcb.leftray.b*fcb.imgwidth + fcb.leftray.m)), (0,255,0),1)
#             cv2.imshow('cb', img2)
#             cv2.waitKey(-1)
            
            if len(forceCombines) == 0: i += 1
            continue
        else:
            i+=1
            continue
    
    #########

    i = 0
    while i < len(allines):
        result = allines[i]
        if result.available:
#             print '------------------------- FAR'
            img2 = illu.copy()
#             result.draw(img2, (125,125,125), 0.5, drawyhat=False)
#             cv2.line(img2, (0, int(result.rightray.m)), (result.imgwidth, \
#                 int(result.rightray.b*result.imgwidth + result.rightray.m)), (255,0,0),1)
#             cv2.line(img2, (0, int(result.leftray.m)), (result.imgwidth, \
#                 int(result.leftray.b*result.imgwidth + result.leftray.m)), (0,255,0),1)
#             cv2.imshow('cb', img2)
#             cv2.waitKey(-1)
            linemap = []
            for j in range(i, len(allines)):
                if j==i: continue
                candidate = allines[j]
                if not candidate.available: continue
                forcedCombined, score = result.scoreLineAfter(candidate)
#                 cv2.waitKey(-1)
                if (not forcedCombined) and score >= 0:
                    linemap.append((score, candidate))            
            if len(linemap) > 0:
                j, candidate = min(linemap)
#                 candidate.draw(img2, str2col(candidate.id), 0.5, drawyhat=False)
#                 cv2.imshow('cb-far', img2)
#                 cv2.waitKey(-1)
                result.combineLinesAfter([candidate])
                continue
            else:
                i+=1
                continue
        else:
            i+=1
            continue

    allines = [line for line in allines if line.available]

    
    print 'DONE LINE, now ILLUSTRATE **************, TOTAL LINE COUNT ' + str(len(allines))
    print 'TOTAL TIME  ' + str(time() -tt)
    img2 = img.copy()
    for line in allines:
        try:
            line._updateCurve()
            col = str2col(line.id)
            line.draw(img2, col, 0.5, drawyhat=False, drawline=True)
        except Exception as e:
            pass
    cv2.imshow('lines-ext', img2)
    cv2.waitKey(-1)
    
    retlines = []
    for line in allines:
        line._updateCurve()
        line.expandPoints()
        imgline = line.extract(img, expands=SubLine.EP_FINAL_EXPAND)
        cv2.imshow('line', imgline)
        cv2.waitKey(-1)
        retlines.append(imgline)
    
    return illu, img2, retlines

    
import random, os
if __name__ == "__main__":
    np.set_printoptions( threshold=np.inf)
    inputpath='/home/loitg/Downloads/complex-bg/special_line/'
    outputpath='/home/loitg/Downloads/complex-bg/java5/'
    filelist = list(os.listdir(inputpath))
    filelist.sort()
    frompos = 0#filelist.index('gimp-temp-1066042.-area.png')
    for filename in filelist[frompos:]:       
        if filename[-3:].upper() == 'PNG':
#         if filename == 'gimp-temp-1066018.-area.png':
            print filename
            illu, img, lines = extractLines2(inputpath + filename)
#             cv2.imwrite(outputpath+filename + '_1.jpg', illu)
#             cv2.imwrite(outputpath+filename + '_4.jpg', img)

#             cv2.imshow('output1', illu)
#             cv2.imshow('output2', img)
#             cv2.waitKey(-1)
