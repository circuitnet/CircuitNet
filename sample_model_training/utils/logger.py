import logging
import datetime
import os 
import sys

def build_logger(arg_dict):
    if arg_dict['test_mode']:
        log_dir = os.path.join(os.path.dirname(arg_dict['pretrained']), 'test-{}'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(log_dir)
        print(log_dir)
        log_file = '{}/test.log'.format(log_dir)
    else:
        log_dir = os.path.join(arg_dict['save_path'], 'train-{}'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        log_file = '{}/train.log'.format(log_dir)
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)) # 重点


    sys.excepthook = handle_exception
    return logger, log_dir