'''
Created on Feb 21, 2018

@author: loitg
'''
class DFO(object):
    pass

args = DFO()
args.model_path = '/home/loitg/debugtf/model_version4_total/'
args.imgsdir = '/home/loitg/Downloads/complex-bg/'
args.numprocess = 1
args.qget_wait_count = 200
args.qget_wait_interval = 0.5
args.bucket_size = 1
args.bucket_max_time = 10
args.device = '/device:CPU:0'
args.javapath = '/home/loitg/location_nn/'
args.dbfile = 'top20_1.csv'
args.locationnjar = 'location_nn.jar'

if __name__ == '__main__':
    pass