'''
Created on Feb 12, 2018

@author: loitg
'''
import os
import sys
import shutil

def recursiveCopy(destination, outpath):
    all_files = []
    first_loop_pass = True
    for root, _dirs, files in os.walk(destination):
        if first_loop_pass:
            first_loop_pass = False
            continue
        for filename in files:
            all_files.append(os.path.join(root, filename))
            try:
                shutil.copyfile(os.path.join(root, filename), os.path.join(outpath, filename))
            except Exception:
                print 'ERROR ' + filename

if __name__ == '__main__':
    try:
        parts = sys.argv[1]
        outpath = sys.argv[2]
    except Exception:
        print("parts, outputfolder")
        sys.exit(0)
    print(parts)
    print(outpath)
    
    recursiveCopy(parts, outpath)
    
