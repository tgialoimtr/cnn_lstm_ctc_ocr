import os, sys
import cv2

if __name__ == "__main__":
    lines = list(open(sys.argv[1]).readlines())
    print len(lines)
    for locode in lines:
        locode = locode.rstrip()
        subdir = os.path.join(sys.argv[2], locode)
        print locode + ':'
        if not os.path.exists(subdir):
            print 'NOT EXIST'
            continue
        for fn in os.listdir(subdir):
            if fn[-3:].upper() != "JPG": continue
            print fn
            img = cv2.imread(os.path.join(subdir, fn))
            cv2.imshow('sample', img)
            k = cv2.waitKey(-1)
            if k == ord('n'): break
            if k == ord('q'): sys.exit(0)
