'''
Created on Feb 12, 2018

@author: loitg
'''
import os
import sys
import shutil

def name2path(destination):
    mapping = {}
    first_loop_pass = True
    for root, _dirs, files in os.walk(destination):
        if first_loop_pass:
            first_loop_pass = False
            continue
        for filename in files:
            mapping[filename] = os.path.join(root, filename)
    return mapping

def recursiveCopy(destination, outpath):
    mapping = name2path(destination)
    for filename, path in mapping.iteritems():
        try:
            shutil.copyfile(path, os.path.join(outpath, filename))
        except Exception:
            print 'ERROR ' + filename
if __name__ == '__main__':
    try:
        parts = sys.argv[1]
        outpath = sys.argv[2]
        selected = sys.argv[3]
    except Exception:
        print("parts, outputfolder, selected")
        sys.exit(0)
    print(parts)
    print(outpath)
    print(selected)
    
    mapping = name2path(parts)
    for i, line in enumerate(open(selected, 'r')):
        fn = line.split(',')[3].rstrip()
        print fn
        if fn not in mapping:
            print 'NOT EXIST'
            continue
        path = mapping[fn]
        print path
        try:
            shutil.copyfile(path, os.path.join(outpath, fn))
        except Exception:
            print 'ERROR ' + fn


# if __name__ == '__main__':
#     try:
#         parts = sys.argv[1]
#         outpath = sys.argv[2]
#     except Exception:
#         print("parts, outputfolder")
#         sys.exit(0)
#     print(parts)
#     print(outpath)
#     
#     recursiveCopy(parts, outpath)
    
