'''
Created on Feb 12, 2018

@author: loitg
'''
import sys, os
from shutil import copyfile

def recursiveCopy(destination, outdest, depth=None):
    if not depth:
        depth = []
    for file_or_dir in os.listdir(os.path.join([destination] + depth, os.sep)):
        if os.path.isfile(file_or_dir):
            copyfile(file_or_dir, outdest)
        else:
            recursiveCopy(destination, outdest, os.path.join(depth + [file_or_dir], os.sep))

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
    
