'''
Created on Nov 3, 2018

@author: loitg
'''
import os, cv2
from test_charboxfind import findBox
from test_lineextract3 import SubLine, estimate_skew_angle, drawPoints
from pylab import *
import ocrolib
from sklearn.externals import joblib
import random

def buildLineRecogData(imgpath, outfile):
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
    yfrom=100; yto=min(img_grey.shape[0], 800);
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


#     illu = img.copy()
#     for bound in objects:
#         cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].start),2, (255,0,0),-1)
#         cv2.circle(illu,((bound[1].start + bound[1].stop)/2, bound[0].stop), 2, (0,255,0),-1)
#         cv2.line(illu, ((bound[1].start + bound[1].stop)/2, bound[0].start), ((bound[1].start + bound[1].stop)/2, bound[0].stop), (0,255,255),1)
#     cv2.imshow('ii', illu)
#     cv2.waitKey(-1)
    random.shuffle(objects)
    for bound in objects: # sorted
        topy = bound[0].start
        boty = bound[0].stop
        x = (bound[1].start + bound[1].stop)/2
        ss = SubLine(topy, boty, x)
        confs = ss.suggest1(None, points, img_grey)
        
        for conf in confs:
            score, confshape, fullscore = ss.score(conf, points)
            
            print '== ' + ss.id + ' score ' + str(conf.angle) + ' ' + str(fullscore)

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
            img5 = img.copy()
            
            try:
                fordraw = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=conf.toppoints,
                                  bottoms=conf.botpoints)
                fordraw.id = ss.id
                fordraw.draw(img5, (0,0,255), 0.5, True)
            except Exception as e:
                print e
            
            drawPoints(img5, conf.toppoints, (255,0,0))
            drawPoints(img5, conf.botpoints, (0,255,0))
            drawPoints(img5, confshape, (0,0,255))
            cv2.imshow('ii', img5)
            k = cv2.waitKey(-1)
            if k == ord('n'):
                return 0
            elif k == ord('1'):
                is_line = True
                is_selected = False
            elif k == ord('2'):
                is_line = True
                is_selected = True
            else: # NOT LINE
                is_line = False
                is_selected = False
            
            
            top_in_0 = len(conf.topsinintervals[0])
            top_in_12 = len(conf.topsinintervals[1]) + len(conf.topsinintervals[2])
            bot_in_12 = len(conf.botsinintervals[1]) + len(conf.botsinintervals[2])
            top_in_45 = len(conf.topsinintervals[4]) + len(conf.topsinintervals[5])
            bot_in_45 = len(conf.botsinintervals[4]) + len(conf.botsinintervals[5])
            bot_in_6 = len(conf.botsinintervals[6])
            
            print '== top_in_0 ', top_in_0, 'top_in_12 ', top_in_12, 'bot_in_12 ', bot_in_12, 'top_in_45 ', top_in_45, 'bot_in_45 ', bot_in_45, 'bot_in_6 ', bot_in_6
            print '== support ' + str(top_in_12) + ', constrast ' + str(np.float64(top_in_0)/top_in_12) + ', outlier ' + str(np.float64(bot_in_12)/top_in_12)
            print '== support ' + str(bot_in_45) + ', constrast ' + str(np.float64(bot_in_6)/bot_in_45) + ', outlier ' + str(np.float64(top_in_45)/bot_in_45)
    
    
            print '== ' + ss.id + ' score ' + str(conf.angle) + ' ' + str(fullscore)
            
            
            print '==' + ','.join([str(x) for x in [ss.id, conf.angle, is_line, top_in_12, is_selected]])
            outfile.write(','.join([str(x) for x in [imgpath, #0
                                                     ss.id, #1
                                                     conf.angle, #2 
                                                     is_line, #3
                                                     top_in_12, #4
                                                     np.float64(top_in_0)/top_in_12, #5 
                                                     np.float64(bot_in_12)/top_in_12, #6
                                                     bot_in_45, #7
                                                     np.float64(bot_in_6)/bot_in_45, #8 
                                                     np.float64(top_in_45)/bot_in_45] + [ #9
                                                         1.0*height/width, #10 
                                                         abs(tanangle)] #11
                                                     + fullscore + # 12,13,14,15,16,17,18,19
                                                     [is_selected] # 20
                                                     
                                                     ]))
            outfile.write('\n')
    
    return img

def readData(path):
    datafile = open(path,'r')
    data = []
    def _toFloat(a):
        if a == 'True':
            return 1
        elif a == 'False':
            return 0
        elif a == 'nan':
            return np.nan
        elif len(a) >=3 and a[-3:] == 'inf':
            return 99
        else:
            return float(a)
        
    for line in datafile.readlines():
        fields = line.rstrip().split(',')
        fields = [_toFloat(x) for x in fields]
        data.append(fields)
        
    return np.array(data)
    
def buildModel1(data):
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    
    clf_names = ["SVM",
                "NN",
                 "DecitionTree"]
    clfs = [SVC(gamma=2, C=1, probability=True),
            MLPClassifier(hidden_layer_sizes=(30,10,5), alpha=1),
           DecisionTreeClassifier(max_depth=4)]
    
    X = data[:, :13]
    print data.shape
    y = (data[:, 14] >0.5) | (data[:, 13] > 0.5)
    for clf, name in zip(clfs, clf_names):
        print "===---==" + name
        for rs in range(30,33):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=rs)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print "--=====" + str(score)
    
    clfs[0].fit(X,y)
    joblib.dump(clfs[0], '/home/loitg/Downloads/complex-bg/le_model_3.pkl')

le_cls_data_path = '/home/loitg/Downloads/complex-bg/le_cls_data_3.csv'

if __name__ == "__main__":
    data = readData(le_cls_data_path)
    data = data[~np.isnan(data).any(axis=1)]
    buildModel1(data)
     
     
    sys.exit(0)

# if __name__ == "__main__":
#     gtdatafile = open(le_cls_data_path, 'w')
#     for filename in os.listdir('/home/loitg/Downloads/complex-bg/special_line/'):
#         if filename[-3:].upper() == 'JPG':
# #         if filename == "4.JPG":
#             print filename
# #             print dir(random)
# #             sys.exit(0)
#             buildLineRecogData('/home/loitg/Downloads/complex-bg/special_line/' + filename, gtdatafile)
            
            
            