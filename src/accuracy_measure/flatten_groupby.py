'''
Created on Feb 12, 2018

@author: loitg
'''
import sys, os
from shutil import copyfile


if __name__ == '__main__':
    try:
        groupbypath = sys.argv[1]
        top600path = sys.argv[2]
        outpath = sys.argv[3]
    except Exception:
        print("groupbypath, topx00.csv, outputfolder");
    print(top600path)
    print(groupbypath)
    
    top600 = {}
    abc = []
    for i, line in enumerate(open(top600path, 'r')):
        temp = line.split(',')
        top600[temp[0]] = {}
        top600[temp[0]]['name'] = temp[1]
        top600[temp[0]]['rank'] = str(i)
        abc.append((i, temp[0]))

    for _, locode in sorted(abc):
        filelist = []
        current_index = 0
        locodepath = os.path.join(groupbypath, locode)
        if not os.path.isdir(locodepath): continue
        i = 0
        for fn in os.listdir(locodepath):
            if fn[-3:].upper() in ['PEG', 'JPG']:
                filelist.append(fn)
                copyfile(os.path.join(locodepath, fn), os.path.join(outpath, '%03d_%s_%d.jpg' % (int(top600[locode]['rank']), locode, i)))
                i += 1
                if i > 2: break 