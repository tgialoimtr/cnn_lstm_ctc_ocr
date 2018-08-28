'''
Common arguments and parameters
'''
class DFO(object):
    pass

args = DFO()
args.model_path = '/home/ntnnhi/ocr-dir/tfmodel/'
args.imgsdir = '/home/ntnnhi/build_data/20180712/image/'
args.textsdir = '/home/ntnnhi/build_data/texts_4A/'
args.numprocess = 1 #CPU:2 #GPU:8
args.qget_wait_count = 40000
args.qget_wait_interval = 0.1
args.bucket_size = 1 #CPU:2 #GPU:16
args.bucket_max_time = 1
args.device = '/device:CPU:0'
args.javapath = '/home/ntnnhi/ocr-dir/java/'
args.logsdir = '/home/ntnnhi/ocr-dir/logs/log_kb'
args.dbfile = 'TOP700_V3_removesemclark.csv'
args.locationnjar = 'location_nn_0825.jar'
args.download_dir = '/home/ntnnhi/ocr-dir/downloads/download/'
args.connection_string = 'DefaultEndpointsProtocol=http;AccountName=storacctcapitastartable;AccountKey=Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA=='
args.queue_get_name = 'ocr-receipt-queue-dev'
args.queue_push_name = 'receipt-info-queue-dev'
args.container_name = 'mobile-receipts-dev'
args.receipt_waiting_interval = 10
args.heartbeat_check = 600
args.mode = 'process-local'
