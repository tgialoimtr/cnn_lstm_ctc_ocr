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
import math
from statsmodels.nonparametric.smoothers_lowess import lowess
from shapely.geometry import Polygon
from scipy.interpolate import interp1d

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
    LOOKAHEAD = 2.5
    ISTOP = 2
    ISBOT = 1
    ISEMPTY = 0
    IGNORED = -1
    LINE_IOU_COMBINE_THRESHOLD = 0.8
    LINE_SCORE_THRESHOLD = 0.1
    
    class Config1(object):
        
        MIN_INSIDER = 2
        MAX_ABOVE_RATIO = 0.2
        MAX_OUTSIDER_RATIO = 0.9
        INTERVALS = [0.2,0.5,0.5]
        MAX_ANGLE = 30.0
        NODE_IN_LINE_THRESHOLD = 0.2
        
        def __init__(self, topy, boty, x, img):
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
            self.height = float(self.aboty - self.atopy)
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
            self.sin = {}; self.cos = {}
            for a in np.linspace(-self.MAX_ANGLE, self.MAX_ANGLE, 9):
                self.sin[a] = math.sin(a/180.0*math.pi)
                self.cos[a] = math.cos(a/180.0*math.pi)
        
        def _add2intervals(self, y1, point, point_type):
            for i, interval in enumerate(self.intervals):
                if y1 >= interval[0] and y1 < interval[1]:
                    if point_type == SubLine.ISTOP:
                        self.topsinintervals[i].append(point)
                    if point_type == SubLine.ISBOT:
                        self.botsinintervals[i].append(point)
        
        def _intervalHasLine(self):
            top_in_0 = len(self.topsinintervals[0])
            top_in_12 = len(self.topsinintervals[1]) + len(self.topsinintervals[2])
            bot_in_12 = len(self.botsinintervals[1]) + len(self.botsinintervals[2])
            top_in_45 = len(self.topsinintervals[4]) + len(self.topsinintervals[5])
            bot_in_45 = len(self.botsinintervals[4]) + len(self.botsinintervals[5])
            bot_in_6 = len(self.botsinintervals[6])
            print 'top_in_0 ', top_in_0, 'top_in_12 ', top_in_12, 'bot_in_12 ', bot_in_12, 'top_in_45 ', top_in_45, 'bot_in_45 ', bot_in_45, 'bot_in_6 ', bot_in_6
            print 'support ' + str(top_in_12) + ', constrast ' + str(np.float64(top_in_0)/top_in_12) + ', outlier ' + str(np.float64(bot_in_12)/top_in_12)
            print 'support ' + str(bot_in_45) + ', constrast ' + str(np.float64(bot_in_6)/bot_in_45) + ', outlier ' + str(np.float64(top_in_45)/bot_in_45)
            if (top_in_12 > self.MIN_INSIDER and 1.0*top_in_0/top_in_12 < self.MAX_ABOVE_RATIO and 1.0*bot_in_12/top_in_12 < self.MAX_OUTSIDER_RATIO) and \
                (bot_in_45 > self.MIN_INSIDER and 1.0*bot_in_6/bot_in_45 < self.MAX_ABOVE_RATIO and 1.0*top_in_45/bot_in_45 < self.MAX_OUTSIDER_RATIO):
                return True
            else:
                return False
            
        def add(self, x,y, point_type):
            if point_type == SubLine.ISTOP:
                self.toppoints.append((x,y))
            elif point_type == SubLine.ISBOT:
                self.botpoints.append((x,y))
            
        def finalize(self):
            print '------------'
            print 'topcount ', len(self.toppoints)
            print 'botcount ', len(self.botpoints)
            img3 = self.img.copy()
            drawTBX(img3, self.atopy, self.aboty, self.ax, (0,0,255))


            rets = []
            for a in np.linspace(-self.MAX_ANGLE, self.MAX_ANGLE, 9):
                img4 = img3.copy()
                self.topsinintervals = [[] for i in range(len(self.intervals))]
                self.botsinintervals = [[] for i in range(len(self.intervals))]
                for point in self.toppoints:
                    x1 = point[0] - self.ax
                    y1 = point[1] - self.atopy                    
                    ytop_proj = y1 + x1 * self.sin[a] / self.cos[a]
                    self._add2intervals(ytop_proj, point, SubLine.ISTOP)
                for point in self.botpoints:
                    x1 = point[0] - self.ax
                    y1 = point[1] - self.atopy                    
                    ybot_proj = y1 + x1 * self.sin[a] / self.cos[a]
                    self._add2intervals(ybot_proj, point, SubLine.ISBOT)
                
                print '--angle ', a
                
                if self._intervalHasLine():
                    conf = SubLine.Config1(self.atopy, self.aboty, self.ax, self.img)
                    conf.angle = a
                    conf.toppoints = self.topsinintervals[1] + self.topsinintervals[2]
                    conf.botpoints = self.botsinintervals[4] + self.botsinintervals[5]
                    rets.append(conf)
                    print 'HAS LINE'
#                     drawPoints(img4, conf.toppoints, (255,0,0))
#                     drawPoints(img4, conf.botpoints, (0,255,0))
#                     cv2.imshow('hasline', img4)
#                     cv2.waitKey(-1)
                else:
                    print 'NO LINE'
#                     drawPoints(img4, self.toppoints, (255,0,0))
#                     drawPoints(img4, self.botpoints, (0,255,0))
                    
#                 cv2.imshow('conf', img4)
#                 cv2.waitKey(-1)
            
            return rets
                
    ###  <<<<<<<<<<<<<<<<------------------
    def __init__(self, topy, boty, x, tops=[], bottoms=[]):      
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

    def _initConfig(self, img):
        possible_confs = []
        possx = self.curx
        posstop = self.curtopy
        possbot = self.curboty
        possible_confs.append(SubLine.Config1(posstop, possbot, possx, img))
#         d = self.height/3
#         possible_confs.append(SubLine.Config1(posstop+d, possbot+d, possx))
#         possible_confs.append(SubLine.Config1(posstop-d, possbot-d, possx))
#         possible_confs.append(SubLine.Config1(posstop+d, possbot+d, possx + d))
####         possible_confs.append(SubLine.Config1(posstop, possbot, possx + d))
#         possible_confs.append(SubLine.Config1(posstop-d, possbot-d, possx + d))
        return possible_confs
    
    def suggest1(self, allnodes, allpoints, img):
        # Generate config
        allconfs = self._initConfig(img)
        retconfs = []
        for x,y in self.nextRange():
            if x >= len(allpoints): continue
            if y >= len(allpoints[x]): continue
            point_type = allpoints[x][y]
            if point_type == self.ISEMPTY: continue
            for cand_conf in allconfs:
                cand_conf.add(x,y,point_type)
        for cand_conf in allconfs:
            confs = cand_conf.finalize()
            if confs is not None and len(confs) > 0:
                retconfs += confs                
        return retconfs
    ###  ------------------>>>>>>>>>>>>>>>>>>
    
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
        return error, leftmostpoint, rightmostpoint
    
    def _intervalScore(self, leftpoint, rightpoint, height, allpoints, linetype):
        int0 = int((SubLine.Config1.INTERVALS[0] + SubLine.Config1.INTERVALS[1])*height)
        int1 = int(SubLine.Config1.INTERVALS[1]*height)
        int2 = 0
        int3 = int(SubLine.Config1.INTERVALS[2]*height)
        alpha = 1.0*(rightpoint[1] - leftpoint[1])/(rightpoint[0] - leftpoint[0])
        wrong_above = 0
        correct_inner = 0
        wrong_inner = 0
        for x in range(leftpoint[0], rightpoint[0] + 1):
            if not (x < len(allpoints)): continue
            y = int(alpha*(x-leftpoint[0]) + leftpoint[1])
            if linetype == SubLine.ISTOP:
                for yy in range(y - int0, y - int1 + 1):
                    if not (yy < len(allpoints[x])): continue
                    if allpoints[x][yy] == SubLine.ISTOP:
                        wrong_above += 1
                for yy in range(y - int1, y + int3 + 1):
                    if not (yy < len(allpoints[x])): continue
                    if allpoints[x][yy] == SubLine.ISTOP:
                        correct_inner += 1
                    elif allpoints[x][yy] == SubLine.ISBOT:
                        wrong_inner += 1
                
            else:
                for yy in range(y + int1, y + int0 + 1):
                    if not (yy < len(allpoints[x])): continue
                    if allpoints[x][yy] == SubLine.ISBOT:
                        wrong_above += 1
                for yy in range(y - int3, y + int1 + 1):
                    if not (yy < len(allpoints[x])): continue
                    if allpoints[x][yy] == SubLine.ISBOT:
                        correct_inner += 1
                    elif allpoints[x][yy] == SubLine.ISTOP:
                        wrong_inner += 1
        print 'wrong_above ', wrong_above
        print 'wrong_inner', wrong_inner
        print 'correct_inner',correct_inner
        return 1.0*wrong_above/(correct_inner+0.1), 1.0*wrong_inner/(correct_inner+0.1)
        
    def score(self, conf, allpoints):
        toperror, topleftpoint, toprightpoint = self._score(self.tops, conf.toppoints, self.lastytopindex)
        topabove, topoutlier = self._intervalScore(topleftpoint, toprightpoint, self.height, allpoints, SubLine.ISTOP)
        boterror, botleftpoint, botrightpoint = self._score(self.bottoms, conf.botpoints, self.lastybotindex)
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
        return retScore, [(xleft, topyleft), (xright, topyright), (xright, botyright), (xleft, botyleft)], [toperror/self.height, topabove, topoutlier,boterror/self.height,botbelow, botoutlier]
#         return retScore, [topleft, topright, botright, botleft]

    def nextRange(self):
        x1 = self.curx
        x2 = x1 + int(self.height * self.LOOKAHEAD * 2)
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
    
    def _filterCombineConf(self, results):
        n = len(results)
        if len(results) > 0: print 'a1 ' + str([result[0] for result in results])
        for i in range(0, n):
            for j in range(i+1, n):
                if results[i] is None or results[j] is None: continue
                iou = self.iou(results[i][2], results[j][2])
                if iou > SubLine.LINE_IOU_COMBINE_THRESHOLD:
                    print 'merge ', results[j][1].angle, results[i][1].angle
                    if results[j][0] < results[i][0]:
                        results[i] = results[j]
                    results[j] = None
        
        results2 = [x for x in results if x is not None]
        del(results)
        results2.sort()
        
        
        
        if len(results2) > 0: 
            print 'a2 ' + str([result[0] for result in results2])
        
            return [results2[0]]
        else:
            return []

    def next(self, allnodes, allpoints, img):
        confs = self.suggest1(allnodes, allpoints, img)
        results = []
        for conf in confs:
            score, confshape, fullscore = self.score(conf, allpoints)
            results.append((score, conf, confshape))
            img4 = img.copy()
            drawPoints(img4, confshape, (0,255,255))
            print self.id + ' score ' + str(conf.angle) + ' ' + str(fullscore)
#             cv2.imshow('hasline', img4)
#             cv2.waitKey(-1)

        selected_confs = self._filterCombineConf(results)
        retlines = []
        for i, (_, conf, confshape) in enumerate(selected_confs):
            
            img5 = img.copy()
            drawPoints(img5, confshape, (255,0,255))
            print self.id + ' final ' + str(self.nextCount)
#             cv2.imshow('final', img5)
#             cv2.waitKey(-1)
            
            topright = confshape[1]
            botright = confshape[2]
            topy = topright[1]; boty = botright[1]; 
            x = (topright[0] + botright[0])/2
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
                                  bottoms=self.bottoms + conf.botpoints)
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
        f2 = interp1d(xs, yhat, kind='cubic')
        return f2
    
    def extractConstHeight(self, expand=0):
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
        
        f_combined = self.smoothedFunc(np.concatenate(tops[:,0], bottoms[:,0]), np.concatenate(tops[:,1]+height, bottoms[:,1]))
        xs = list(range(x1,x4))
        y_bot = [f_combined(x) for x in xs]
        y_top = [y - height for y in y_bot]
        
        return xs, y_top, y_bot

    
    def draw(self, img, col, opacity, drawyhat=True):
        self.nextCount
        self.id
        
        xs, y_top, y_bot = self.extractConstHeight()
        
        temp = np.zeros_like(img)
        for i in range(len(xs)):
            temp[y_top[i]:y_bot[i],xs[i]] = col
        
        cv2.addWeighted(img, 1-opacity, temp, opacity, img)
        if drawyhat:
            drawPoints(img, zip(xs[::4], y_top[::4]), col)
            drawPoints(img, zip(xs[::4], y_bot[::4]), col)

        cv2.putText(img,self.id, (xs[0],y_bot[0]), cv2.FONT_HERSHEY_SIMPLEX, 2, col)
        return temp
        
    
    
    def extract(self, img, expand=0):
        xs, y_top, y_bot = self.extractConstHeight()
        height = y_bot[0] - y_top[0]
        n = len(xs)
        retline = np.zeros((height,n))
        for i in range(n):
            retline[:,i] = img[y_top[i]:y_bot[i],xs[i]]
        return retline
        
    def clear(self, allnodes, clearedList):
        xs, y_top, y_bot = self.extractConstHeight()
        height = y_bot[0] - y_top[0]
        threshold = int(height*SubLine.NODE_IN_LINE_THRESHOLD)
        for i in range(len(xs)):
            x = xs[i]
            for y in range(y_bot[i] - threshold, y_bot[i] + threshold):
                node = allnodes[x][y]
                if node is None:
                    pass
                elif  y_top[i] - threshold < node[0] < y_top[i] + threshold:   
                    clearedList.add(node)    
    
class Abc(object):
    def __init__(self, img_grey, illu_scale):
        self.img_grey = img_grey
        self.illu = cv2.cvtColor(img_grey.astype(np.float32), cv2.COLOR_GRAY2BGR)
        self.illu = cv2.resize(self.illu, None, fx=illu_scale, fy=illu_scale)
        self.illu = (self.illu*255).astype(np.uint8)             

    def drawLine(self, subline):
        subline.param *= 2
        subline.draw(self.illu)
    
    def getIllu(self):
        return self.illu
    
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
    xfrom=100; xto=500;
    yfrom=800; yto=1080;
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
    
    illustrator = Abc(img_grey, 2.0)
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

    illu = img.copy()
    for bound in objects:
        cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].start),2, (255,0,0),-1)
        cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].stop), 2, (0,255,0),-1)
        cv2.line(illu, ((bound[1].start + bound[1].stop)/2, bound[0].start), ((bound[1].start + bound[1].stop)/2, bound[0].stop), (0,0,255),1)
    cv2.imshow('ii', illu)

    for bound in objects: # sorted
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        if (topy, boty, x) in clearedList: continue
        subline = SubLine(topy=topy,
                          boty=boty,
                          x=x)
        allines.append(subline)
        subline.isnew = False
        move(subline, nodes, points,img)
    
    for line in allines:
        illustrator.draw(line)
    
    
    return img, illustrator.getIllu()

    
import os
if __name__ == "__main__":
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/tmp/'):        
#         if filename[-3:].upper() == 'JPG':
        if filename == '34.JPG':
            print filename
            img, illu = extractLines2('/home/loitg/Downloads/complex-bg/tmp/' + filename)
            cv2.imshow('illu', illu)
            cv2.waitKey(-1)

