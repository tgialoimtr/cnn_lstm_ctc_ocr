#!/usr/bin/env python
import sys
from asn1crypto.x509 import Features
from click.termui import clear
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

def linfit(xs, ys):
    n = len(xs)
    sumx = 0; sumy = 0; sumxy = 0; sumx2 = 0; sumy2 = 0; 
    for i in range(n):
        xi = xs[i]; yi = ys[i]
        sumx += xi
        sumy += yi
        sumx2 += xi*xi
        sumy2 += yi*yi
        sumxy += xi*yi
    denom = (n*sumx2 - sumx*sumx)
    
    b = 1.0* (n*sumxy - sumx*sumy) / denom
    m = 1.0* (sumy*sumx2 - sumx*sumxy) /denom
    s2e = (n*sumy2 - sumy*sumy - b*b*denom)/(n*(n-2))
    
    return b,m,s2e

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
    INTERVAL_BASIC_LEN = [0.2,0.5]
    MAX_ANGLE = 30.0
    NUM_ANGLE = 31

    def _resetCur(self, topy, boty, x):
        self.curtopy = topy
        self.curboty = boty
        self.height = float(self.aboty - self.atopy)
        self.curx = x
        self.curnextx = self.curx + int(self.height * self.LOOKAHEAD * 2)
        ### 
        self.heightchkpts = []
        self.heightchkpts.append(-self.height*(self.INTERVALS[0] + self.INTERVALS[1]))#int0
        self.heightchkpts.append(-self.height*(self.INTERVALS[1]))#int1
        self.heightchkpts.append(0.0)#int2
        self.heightchkpts.append(self.INTERVALS[2] * self.height)#int3
        self.heightchkpts.append(self.height - self.INTERVALS[2] * self.height)#int4
        self.heightchkpts.append(self.height)#int5
        self.heightchkpts.append(self.height + self.height*self.INTERVALS[1])#int6
        self.heightchkpts.append(self.height + self.height*(self.INTERVALS[1] + self.INTERVALS[0]))#int7
        self.intervals = [(self.heightchkpts[i], self.heightchkpts[i+1]) for i in range(0,7)] # 012--456
        self.topsinintervals = [[] for i in range(len(self.intervals))]
        self.botsinintervals = [[] for i in range(len(self.intervals))]
    
    
    ###  <<<<<<<<<<<<<<<<------------------
    def __init__(self, topy, boty, x, tops=[], bottoms=[], lemodel=None):      
        self.tops = tops
        self.bottoms = bottoms
        self.curtopy = int(topy)
        self.curboty = int(boty)
        self.height = boty-topy
        self.curx = int(x)
        self.isnew = True
        self.nextCount = 0
        self.id = str(self.curboty) + '.' + str(self.curx)
        self.lastytopindex = 0
        self.lastybotindex = 0
        self.lemodel = lemodel
        
        
        self.img = img
        # pre-set
        self.atopy=int(topy)
        self.aboty=int(boty)
        self.ax=int(x)
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

    ###  ------------------>>>>>>>>>>>>>>>>>>
    
    def _uniformScore(self, xs):
        N = len(xs)
        if N < 3:
            return 0.0
        else:
            L = max(xs) - min(xs)
            ds = [xs[i+1] - xs[i] for i in range(0,N-1)]
            a = 1.0*L/(N-1)
            b = np.median(ds)
            return abs(a-b)/a
    
    def _score(self, lefts, rights, fromindexleft=0):
        if lefts is not None and len(lefts) > 0:
            lefts = np.array(lefts)
            rights = np.array(rights)
            xs = np.concatenate((lefts[fromindexleft:,0], rights[:,0]))
            ys = np.concatenate((lefts[fromindexleft:,1], rights[:,1]))
        else:
            rights = np.array(rights)
            xs = rights[:,0]
            ys = rights[:,1]
        _, indices = np.unique(xs, return_index=True) 
        xs = xs[indices]
        ys = ys[indices]
#         yhat = lowess(ys, xs, frac=0.666, is_sorted=False, return_sorted=False, delta=0.0)
#         error = np.sum(np.square((yhat - ys)/self.height))/len(ys)
#         leftmostpoint = (xs[0], yhat[0])
#         rightmostpoint = (xs[-1], yhat[-1])
        
        b,m,error = linfit(xs,ys)
        leftmostpoint = (xs[0], xs[0]*b+m)
        rightmostpoint = (xs[-1], xs[-1]*b+m)
        uniformity = self._uniformScore(xs)
        return error, leftmostpoint, rightmostpoint, uniformity
    
    def _intervalScore(self, leftpoint, rightpoint, height, allpoints, linetype):
        int0 = int((SubLine.Config1.INTERVALS[0] + SubLine.Config1.INTERVALS[1])*height)
        int1 = int(SubLine.Config1.INTERVALS[1]*height)
        int2 = 0
        int3 = int(SubLine.Config1.INTERVALS[2]*height)
        alpha = 1.0*(rightpoint[1] - leftpoint[1])/(rightpoint[0] - leftpoint[0])
        
        return 1.0*wrong_above/(correct_inner+0.1), 1.0*wrong_inner/(correct_inner+0.1)
        
    def score(self, conf, allpoints):
        toperror, topleftpoint, toprightpoint, topuniform = self._score(self.tops, conf.toppoints, self.lastytopindex)
        topabove, topoutlier = self._intervalScore(topleftpoint, toprightpoint, self.height, allpoints, SubLine.ISTOP)
        boterror, botleftpoint, botrightpoint, botuniform = self._score(self.bottoms, conf.botpoints, self.lastybotindex)
        botbelow, botoutlier = self._intervalScore(botleftpoint, botrightpoint, self.height, allpoints, SubLine.ISBOT)
        
        ## how to combine multiple features to one score
        retScore = toperror/self.height + boterror/self.height + topabove + topoutlier + botbelow + botoutlier
        
        ## 
        xleft = (topleftpoint[0] + botleftpoint[0])/2
        topyleft = topleftpoint[1]
        botyleft = botleftpoint[1]
        xright = (toprightpoint[0] + botrightpoint[0])/2
        topyright = toprightpoint[1]
        botyright = botrightpoint[1]
        return retScore, [(xleft, topyleft), (xright, topyright), (xright, botyright), (xleft, botyleft)], [toperror/self.height, topabove, topoutlier, topuniform, \
                                                                                                            boterror/self.height,botbelow, botoutlier, botuniform]
#         return retScore, [topleft, topright, botright, botleft]

    def nextRange(self):
        x1 = self.curx
        x2 = self.curnextx
        for x in range(x1,x2):
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
    
    
    
    def _filterCombineConf2(self, results):
        results2 = []
        state = 'add'
        for result in results:
            fullscore, conf, confshape = result
            isLine, width, angle = self.lemodel.predict(confshape, fullscore)
            if isLine:
                if state == 'add':
                    results2.append((conf, width, confshape))
                    state = 'update'
                elif state == 'update':
                    lastconfwidth = results2[-1][1]
                    if width > lastconfwidth:
                        results2[-1] = (conf, width, confshape)
            else:
                state = 'add'
            
        return [(x[1],x[0],x[2]) for x in results2]
    
    def _a2i(self, a):
        return i
    
    def _toPolar(self, origin, point):
        angle_list = list(np.linspace(-self.MAX_ANGLE, +self.MAX_ANGLE, self.NUM_ANGLE+1))
        delta_x = point[0] - origin[0]
        delta_y = point[1] - origin[1]
        
        delta_x 
        
        return alpha, x
    def buildAngleFeatures(self, ax_topbotABCD, ax_ydiff):
        self.height
        angle_list
        f1 = 
        
        ### filter rows enough topB and botC counts only
        features = np.zeros((), dtype=float)
        features[:,0] = ...
        
        ax_topmask = np.cumsum(ax_topB) # reverse
        
        ax_botmask = np.cumsum(ax_botC) # reverse
        
        ax_mask = ax_topmask & ax_botmask
        widths = np.sum(ax_mask, axis=0)
        ### check widths > 0
        
        ax_topA = ax_topA & ax_mask
        ... x5
        
        count_topB = np.sum(ax_topB, axis=0)
        count_botC = np.sum(ax_topB, axis=0)
        
        ### check nan and inf
        
        f2 = count_topA/count_topB
        ...
        
        
        f7 = 1.0*self.height/widths
        
        f8 = np.sum(ax_ydiff**2, axis=0)
        
        
        def my_func(row):
            indices = np.where(row)[0].tolist()
            indices[i+1] - indices[i]
        
        f9 = np.apply_along_axis(my_func, 0, ax_topB)
        f10 = np.apply_along_axis(my_func, 0, ax_botC)
        
        
        return features


self.ax_y = ...
self.ax_ydiff = ...
self.clf = joblib.load(modelpath) 

    def next2(self, allnodes, allpoints, img):
        retconfs = []
        
        ax_topA = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topB = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_topC = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botB = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botC = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        ax_botD = np.array((self.NUM_ANGLE, self.curnextx - self.curx), dtype=np.bool)
        
        for x,y in self.nextRange():
            if x >= len(allpoints): continue
            if y >= len(allpoints[x]): continue
            point_type = allpoints[x][y]
            if point_type == self.ISEMPTY: continue
            
            alpha_x_s = [self._toPolar(chkpoint, (x,y)) for chkpoint in chkpoints]
            delta_x = alpha_x_s[0][1]
            alpha_i_s = [self._a2i(alpha_x[0]) for alpha_x in alpha_x_s]
            
            if point_type == self.ISTOP:
                ax_topA[alpha_i_s[0]:alpha_i_s[1], delta_x] = True
                ax_topB[alpha_i_s[1]:alpha_i_s[2], delta_x] = True
                ax_topC[alpha_i_s[2]:alpha_i_s[3], delta_x] = True
            elif point_type == self.ISBOT:
                ax_botB[alpha_i_s[1]:alpha_i_s[2], delta_x] = True
                ax_botC[alpha_i_s[2]:alpha_i_s[3], delta_x] = True
                ax_botD[alpha_i_s[3]:alpha_i_s[4], delta_x] = True
            
            ax_ydiff[alpha_i_s[0]:alpha_i_s[1], delta_x] = chkpoints[0][1]
            ax_ydiff[alpha_i_s[0]:alpha_i_s[1], delta_x] = chkpoints[0][1]
            ax_ydiff[alpha_i_s[0]:alpha_i_s[1], delta_x] = chkpoints[0][1]
            ax_ydiff[alpha_i_s[0]:alpha_i_s[1], delta_x] = chkpoints[0][1]
                    
        features = self.buildAngleFeatures(ax_topbotABCD, ax_ydiff)
        
        results = self.clf.predict(features)
        
        self._filterCombineConf3()
        
        
        
        

        => angles for next:
        
        
        for angle in angles:
            linfit
            
            topy = ...
            boty = ...
            x = ...
            
            
            
            
            clear
            
            if i == len(selected_confs)-1:
                self.curtopy = int(topy)
                self.curboty = int(boty)
                self.curx = int(x)
                self.lastytopindex = len(self.tops)
                self.lastybotindex = len(self.bottoms)
                self.tops = self.tops + conf.toppoints
                self.bottoms = self.bottoms + conf.botpoints
                self.nextCount += 1
                retlines.append(self)
            else:
                another = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=self.tops + conf.toppoints,
                                  bottoms=self.bottoms + conf.botpoints,
                                  lemodel=self.lemodel)
                another.lastytopindex = len(self.tops)
                another.lastybotindex = len(self.bottoms)  
                another.isnew = True
                another.nextCount = self.nextCount + 1
                another.id = self.id
                retlines.append(another)

            
        return sublines
        


    def next(self, allnodes, allpoints, img):
        confs = self.suggest1(allnodes, allpoints, img)
        results = []
        for conf in confs:
            try:
                score, confshape, fullscore = self.score(conf, allpoints)
                results.append((fullscore, conf, confshape))
            except Exception as e:
                print e
                pass
#             img4 = img.copy()
#             drawPoints(img4, confshape, (0,255,255))
            print self.id + ' score ' + str(conf.angle) + ' ' + str(fullscore)
#             cv2.imshow('hasline' + str(conf.angle), img4)
#             cv2.waitKey(-1)

        selected_confs = self._filterCombineConf2(results)
        print 'JFOIDSJFOSDJFPFDSJF ' + str(len(selected_confs))
        retlines = []
        for i, (_, conf, confshape) in enumerate(selected_confs):          
            topright = confshape[1]
            botright = confshape[2]
            topy = topright[1]; boty = botright[1]; 
            x = min(topright[0], botright[0])
            
#             img5 = img.copy()
#             fordraw = SubLine(topy=topy,
#                               boty=boty,
#                               x=x,
#                               tops=self.tops + conf.toppoints,
#                               bottoms=self.bottoms + conf.botpoints,
#                               lemodel=self.lemodel)
#             fordraw.id = self.id
#             fordraw.draw(img5, (255,0,255), 0.5, True)
            print self.id + ' final ' + str(self.nextCount)
#             cv2.imshow('final', img5)
#             cv2.waitKey(-1)
            
            if i == len(selected_confs)-1:
                self.curtopy = int(topy)
                self.curboty = int(boty)
                self.curx = int(x)
                self.lastytopindex = len(self.tops)
                self.lastybotindex = len(self.bottoms)
                self.tops = self.tops + conf.toppoints
                self.bottoms = self.bottoms + conf.botpoints
                self.nextCount += 1
                retlines.append(self)
            else:
                another = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=self.tops + conf.toppoints,
                                  bottoms=self.bottoms + conf.botpoints,
                                  lemodel=self.lemodel)
                another.lastytopindex = len(self.tops)
                another.lastybotindex = len(self.bottoms)  
                another.isnew = True
                another.nextCount = self.nextCount + 1
                another.id = self.id
                retlines.append(another)
        
#             for point in conf.toppoints: 
#                 allpoints[point[0]][point[1]] = SubLine.IGNORED
#             for point in conf.botpoints: 
#                 allpoints[point[0]][point[1]] = SubLine.IGNORED
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
        
    def clear(self, allnodes, clearedList):
        xs, y_top, y_bot = self.extractConstHeight()
        if len(xs) == 0: return
        height = y_bot[0] - y_top[0]
        threshold = int(height*SubLine.NODE_IN_LINE_THRESHOLD)
        for i in range(len(xs)):
            x = xs[i]
            if x >= len(allnodes): continue
            for y in range(y_bot[i] - threshold, y_bot[i] + threshold):
                if y >= len(allnodes[x]) or allnodes[x][y] is None: continue
                node = allnodes[x][y]
                if  y_top[i] - threshold <= node[0] < y_top[i] + threshold:  
                    clearedList.add(node)    

    
class LeModelChooseLine(object):
    def __init__(self, modelpath, clf_type='SVM'):
        self.clf = joblib.load(modelpath)   

    def predict(self, confshape, fullscore):
        topleft = confshape[0]
        topright = confshape[1]
        botright = confshape[2]
        botleft = confshape[3]
        
        topy = topright[1]; boty = botright[1]; 
        x = (topright[0] + botright[0])/2; assert(topright[0] == botright[0])
        xleft = (topleft[0] + botleft[0])/2;
        yleft_bot = botleft[1]
        height = boty - topy
        width = x - xleft
        tanangle = (boty - yleft_bot)/(x - xleft)
        angle = math.atan2((boty - yleft_bot), (x - xleft))/math.pi*180.0
        features = [1.0*height/width, abs(tanangle)] + fullscore
        if np.isnan(features).any() or width < 1:
            return False, 0,0
        else:
            print features
            features = np.array(features).reshape(1,len(features))
            try:
                result = self.clf.predict(features)
            except Exception as e:
                return False, 0,0
            return (result[0] > 0.5), 1.0*width/height, angle

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
    
    lemodel = LeModelChooseLine('/home/loitg/Downloads/complex-bg/le_model.pkl')
    
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
                          lemodel=lemodel)
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
    filelist = list(os.listdir('/home/loitg/Downloads/complex-bg/java/'))
    random.shuffle(filelist)
    for filename in filelist:       
#         if filename[-3:].upper() == 'JPG':
        if filename == '12a.JPG':
            print filename
            img = extractLines2('/home/loitg/Downloads/complex-bg/java/' + filename)
#             cv2.imwrite('/home/loitg/Downloads/complex-bg/java4/'+filename, img)
