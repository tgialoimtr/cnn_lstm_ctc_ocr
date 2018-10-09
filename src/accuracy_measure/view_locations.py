import os, sys
import cv2
from shutil import copyfile

# if __name__ == "__main__":
#     lines = list(open(sys.argv[1]).readlines())
#     print len(lines)
#     for locode in lines:
#         locode = locode.rstrip()
#         subdir = os.path.join(sys.argv[2], locode)
#         print locode + ':'
#         if not os.path.exists(subdir):
#             print 'NOT EXIST'
#             continue
#         for fn in os.listdir(subdir):
#             if fn[-3:].upper() != "JPG": continue
#             print fn
#             img = cv2.imread(os.path.join(subdir, fn))
#             cv2.imshow('sample', img)
#             k = cv2.waitKey(-1)
#             if k == ord('n'): break
#             if k == ord('q'): sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'listimg, container, (destination)'
        sys.exit(0)
    txt = False
    lines = list(open(sys.argv[1]).readlines())
    print len(lines)
    for i, imgname in enumerate(lines):
        imgname = imgname.rstrip()
        if len(imgname) < 3 or imgname[-3:].upper() not in ['TXT','JPG']: continue
        imgpath = os.path.join(sys.argv[2], imgname)
        if txt:
            imgpath += '.txt'
            print '---------' + imgname + '------------'
            print imgpath
            for line in open(imgpath, 'r').readlines():
                print line.rstrip()
        else:
            copyfile(imgpath, os.path.join(sys.argv[3], '%03d_%s' % (i, imgname)))
