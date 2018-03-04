'''
Common arguments and parameters
'''
class DFO(object):
    pass

args = DFO()
args.model_path = '/home/loitg/debugtf/model_version4_total/'
args.imgsdir = '/home/loitg/Downloads/complex-bg/'
args.numprocess = 2
args.qget_wait_count = 200
args.qget_wait_interval = 0.5
args.bucket_size = 1
args.bucket_max_time = 10
args.device = '/device:CPU:0'
args.javapath = '/home/loitg/location_nn/'
args.dbfile = 'top20_1.csv'
args.locationnjar = 'location_nn.jar'
args.download_dir = '/home/loitg/location_nn/downloads'
args.connection_string = 'DefaultEndpointsProtocol=http;AccountName=storacctcapitastartable;AccountKey=Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA=='
args.queue_get_name = 'loitg-queue-get'
args.queue_push_name = 'loitg-queue-push'
args.container_name = 'loitg-local'
args.receipt_waiting_interval = 10 #seconds
args.heartbeat_check = 60 #second
args.mode = 'show' # delete, upload, process