#!/usr/bin/env python
import sys
from pickle import BINSTRING
sys.path += ['/usr/local/lib/python2.7/dist-packages/mininet-2.2.1-py2.7.egg', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/loitg/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0']
import cv2
from pylab import *
from scipy.ndimage import morphology
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, minimum_filter
from scipy.ndimage import measurements
import ocrolib
from skimage.filters import threshold_sauvola

from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import os
import random

from collections import Counter
from time import time
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
args.connect = 2
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

class SauvolaTree(object):
    def __init__(self):
        self.nodes = {}
        self.nodesPerLayer = []
        self.labels = None
        self.bins = None
        self.scales = []

    def createNode(self, key, **kwargs):
        assert len(key) == 2
        node = SauvolaNode(key, **kwargs)
        self.nodes[key] = node
        return node

    def __getitem__(self, key):
        assert len(key) == 2
        return self.nodes[key]

    def __setitem__(self, key, value):
        assert len(key) == 2
        self.nodes[key] = value
        return self.nodes[key]
    
    def addRelation(self, key_parent, key_child):
        self.nodes[key_parent].addChild(self.nodes[key_child])

class SauvolaNode(object):
    
    def __init__(self, key, **kwargs):
        assert len(key) == 2
        self.key = key
        self.state = 'ignore' #keep, ignore, merged
        self.parent = None
        self.children = []
        self.prop = kwargs
    
    def addChild(self, node):
        node.parent = self
        self.children.append(node)
    
    def addParent(self, parent):
        parent.addChild(self)
        
def counter(l):
    l.sort()
    pi = l[0]; ret = {}; count = 0
    for i in l:
        if i != pi:
            ret[i] = count
            pi = i
            count = 1
        else:
            count += 1
    ret[i] = count
    return ret

def buildNextLayerRelation(l1, l2):
    l1 = l1.ravel()
    l2= l2.ravel()
    l = 10000*l1 + l2
    l.sort()
    pi = l[0]; ret = {}; count = 0
    for i in l:
        if i != pi:
            ret[(i/10000, i%10000)] = count
            pi = i
            count = 1
        else:
            count += 1
            
    ret[(i/10000, i%10000)] = count
    return ret

def sauvolatree(img_grey):
    (h, w) = img_grey.shape[:2]

    img_grey = minimum_filter(img_grey, size=(3, 1))
    img_grey = maximum_filter(img_grey, size=(3, 1))
    

#     d = pd.DataFrame(columns=list(range(35)))
    stree = SauvolaTree()
    stree.createNode((-1,-1), area=np.inf, center=(w/2,h/2), bound=(slice(0,h,None), slice(0,w,None)))
    labels = {}
    bins = {}
    previousid = -1
    scalelist = []
    for layerid, k in enumerate(np.linspace(0.01,0.52,10)):
        print 'at ', layerid, k
        binary = sauvola(img_grey, w=30, k=k, scaledown=0.2, reverse=True)
        bins[layerid] = binary
        label, n = measurements.label(binary)
        stree.nodesPerLayer.append(n)
        labels[layerid] = label
        objects = measurements.find_objects(label)
        print 'creating ', n, ' nodes'
        tt = time()
        for i in range(n):
            stree.createNode((layerid, i+1), area=0, bound=objects[i], mask=None) #((label==(i+1))[objects[i]]).astype(np.uint8))
        print 'create node ', time() - tt; tt = time()
        if previousid == -1:
            for i in range(n):
                stree.addRelation((-1,-1),(layerid,i+1))
            print 'add relation ', time() - tt; tt = time()
        else:
            relations = buildNextLayerRelation(labels[previousid], label)
            print 'build relation ', time() - tt; tt = time()
            for mapping, count in relations.iteritems():
                label1, label2 = mapping
                if label1 == 0 and label2 > 0:
                    print 'inversion, abnormal'
                if label1 > 0 and label2 > 0:
                    stree[(layerid, label2)].prop['area'] = count
                    stree.addRelation((previousid,label1),(layerid,label2))
 
                    bound = stree[(layerid,label2)].prop['bound']
                    height = bound[0].stop - bound[0].start
                    
                    previousbound = stree[(previousid,label1)].prop['bound']
                    heightratio = 1.0*(bound[0].stop - bound[0].start)/(previousbound[0].stop - previousbound[0].start) if previousbound[0].stop - previousbound[0].start > 0 else 0
                    if heightratio > 0.85 and height > 4 and layerid < 4:
                        scalelist.append(height)
                    
#                     if heightratio < 0.85:
#                         cv2.rectangle(illu, (bound[1].start*6, bound[0].start*6),(bound[1].stop*6, bound[0].stop*6), (255,0,0),1)
#                         cv2.putText(illu,str(round(heightratio,2)),(bound[1].start*6, bound[0].stop*6), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),1)
#                     else:
#                         cv2.rectangle(illu, (bound[1].start*6, bound[0].start*6),(bound[1].stop*6, bound[0].stop*6), (0,0,255),1)
            print 'add relation ', time() - tt; tt = time()
            
            
#         a = dict(Counter(scalelist))
#         d = d.append(a, ignore_index=True)
#         print a
#             illu = resizeToHeight(illu, 900)
            
#             cv2.imwrite('/home/loitg/Downloads/complex-bg/java/'+ imgpath.split('/')[-1] + '_'+ str(layerid) + '.jpg', illu)
        
        
        previousid = layerid
#     d.to_csv('/home/loitg/Downloads/complex-bg/java/' + imgpath.split('/')[-1] + '.txt')
    freq = counter(scalelist)
    scalecount = [0]*60; n = 0
    for c, v in freq.iteritems():
        if c < len(scalecount):
            scalecount[c] = v
            n += 1
    mean_cand_scale = 1.0 * sum(scalecount) / n
    print 'OOOII----------- ', mean_cand_scale
    peaks = []
    for i in range(5,len(scalecount)-4):
        m = max(scalecount[i], scalecount[i+1], scalecount[i+2], scalecount[i+3], scalecount[i+4])
        if scalecount[i+2] >= m and scalecount[i+2] > mean_cand_scale*1.3:
            peaks.append(i+2)
    if len(peaks) == 0:
        for i in range(6,len(scalecount)-2):
            m = max(scalecount[i], scalecount[i+1], scalecount[i+2])
            if scalecount[i+1] >= m and scalecount[i+1] > mean_cand_scale*1.3:
                peaks.append(i+1)
    print 'OOOII----------- ', scalecount
    print 'OOOII----------- ', peaks
    stree.labels = labels
    stree.bins = bins
    stree.scales = peaks
    return stree

def same(parentshape, childshape):
    dx = parentshape[0] - childshape[0]
    dy = parentshape[1] - childshape[1]
    return abs(dx) <= 2 and abs(dy) <= 2

def extendRange(a, b, d, r):
    return 6, int(b*r)
    

def traverseEditState(node, scalemin, scalemax):
    bound = node.prop['bound']
    height = bound[0].stop - bound[0].start
    if height < scalemin:
        return
    elif height > scalemax:
        for child in node.children:
            traverseEditState(child, scalemin, scalemax)
    else:
        width = bound[1].stop - bound[1].start
        for child in node.children:
            child_bound = child.prop['bound']
            child_width = child_bound[1].stop - child_bound[1].start
            child_height = child_bound[0].stop - child_bound[0].start
            if same((width, height), (child_width, child_height)):
                if node.state == 'ignore': node.state = 'keep'
                child.state = 'merged'
                traverseEditState(child, scalemin, scalemax)
                return 
            elif child_height < scalemin:
                continue
            elif child_height > scalemax:
                continue
            else:
                traverseEditState(child, scalemin, scalemax)

                
def flattenByKeepState(stree):
    bins = stree.bins
    labels = stree.labels
    scales = stree.scales
    ret = []
    for layerid in range(len(bins)):
        binary = bins[layerid]
        label = labels[layerid]
        for i in range(stree.nodesPerLayer[layerid]):
            parent = stree[(layerid, i+1)]
            if parent.state != 'keep':
                continue
            parent_bound = parent.prop['bound']
            ret.append(parent_bound)
    return ret

def randomColor():
    return (random.randint(50,255) ,random.randint(50,255),random.randint(50,255))

def findBox(img):
    if len(img.shape) >= 3 and img.shape[2] > 1:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_grey = img
    stree = sauvolatree(img_grey)
    scalemin, scalemax = extendRange(min(stree.scales), max(stree.scales), 3, 3.5)
    traverseEditState(stree[(-1,-1)], scalemin, scalemax)
    objects = flattenByKeepState(stree)
    return objects, np.mean(stree.scales)
            
            
if __name__ == "__main__":
#     imgpath = '/home/loitg/Downloads/complex-bg/special_line/'
    imgpath = '/home/loitg/Downloads/complex-bg/tmp/'
    for filename in os.listdir(imgpath):        
        if filename[-3:].upper() == 'JPG':
            print filename
            img_grey = ocrolib.read_image_gray(imgpath + filename)
            stree = sauvolatree(img_grey)
            if len(stree.scales) == 0:continue
            scalemin, scalemax = extendRange(min(stree.scales), max(stree.scales), 3, 3.5)
            traverseEditState(stree[(-1,-1)], scalemin, scalemax)
            objects = flattenByKeepState(stree)
            illu = cv2.cvtColor(stree.bins[1]*255, cv2.COLOR_GRAY2BGR)
            illu = cv2.resize(illu, None, fx=6.0, fy=6.0)
            for bound in objects:
                cv2.circle(illu,(bound[1].start*3 + bound[1].stop*3, bound[0].start*6),3, (0,0,255),-1)
                cv2.circle(illu,(bound[1].start*3 + bound[1].stop*3, bound[0].stop*6), 3, (0,255,0),-1)
                cv2.line(illu, (bound[1].start*3 + bound[1].stop*3, bound[0].start*6), (bound[1].start*3 + bound[1].stop*3, bound[0].stop*6), (255,0,0),2)
#                 cv2.rectangle(illu, (bound[1].start*6, bound[0].start*6),(bound[1].stop*6, bound[0].stop*6), randomColor() ,1)
            cv2.imwrite('/home/loitg/Downloads/complex-bg/java32/'+filename + '.jpg', illu)

#             cv2.imshow('illu', illu)
#             cv2.waitKey(-1)

