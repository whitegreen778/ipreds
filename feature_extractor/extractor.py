import os
from pathlib import Path
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
import pickle
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import scipy.sparse
import time
import json
from UltraDict import UltraDict as UD
from tqdm import tqdm
import pickle
import os
import random
import feature_extractor.utils as utils
import math

import warnings
warnings.filterwarnings("ignore")

def adder(app_feature_num: int, row:int):
    global task_queue, result_queue, row_count
    
    matrix = scipy.sparse.csr_matrix((row, app_feature_num), dtype=int)
    m_list = []
    while True:
        item = task_queue.get()
        if item is not None:
            #matrix += item
            m_list.append(item)
        else:
            for m in m_list:
                matrix += item
            result_queue.put(matrix)
            result_queue.put(None)
            break

def args_wrapper_adder(args):
    return add_app_matrix(*args)

def add_app_matrix(dirs:Path, app_feature_num:int, row:int):
    matrix = scipy.sparse.csr_matrix((row, app_feature_num), dtype=int)
    for subnet in dirs:
        matrix += scipy.sparse.load_npz(subnet / utils.APP_MATRIX_FILENAME)
    return matrix


class ExtractorOfFile(ABC):

    def __init__(self, dir, processes=(os.cpu_count() - 2)):
        self.dir = Path(dir)
        (self.dir / 'subnet').mkdir(exist_ok=True, parents=True)
        self.processes = processes

    def known_subent_extract(self, using_app_feature:set, need_port_count=True, need_app_num_dict=True):
        self.using_app_feature = using_app_feature
        self.need_port_count = need_port_count
        self.need_app_num_dict = need_app_num_dict

        if need_app_num_dict or need_port_count:
            args = []
            for subnet_dir in (self.dir / 'subnet').iterdir():
                if subnet_dir.is_dir():
                    args.append(subnet_dir)
            
            pbar = tqdm(total=len(args), ncols=80, ascii=True)
            pool = multiprocessing.Pool(processes=self.processes)
            app_set = set()

            for result in pool.imap_unordered(self.subnet_extract, args, chunksize=25):
                if need_app_num_dict:
                    app_set.update(result)

                pbar.update()
            
            if need_app_num_dict:
                app_num_dict = {}
                for i, app in enumerate(app_set):
                    app_num_dict[app] = i + 1
                
                with open(self.dir / utils.APP_NUM_DICT_FILENAME, 'wb') as f:
                    pickle.dump(app_num_dict, f)



class FeatureSaver:
    def __init__(self, dir:Path, file_extractor:ExtractorOfFile, 
                subnet_threads=4, chunk_size=25):
        self.dir = Path(dir)
        self.file_extractor = file_extractor
        (self.dir / 'subnet').mkdir(exist_ok=True, parents=True)
        self.subnet_threads = subnet_threads
        self.chunk_size = chunk_size



    def arg_wrapper_calc_matrix(self, args):
        return self.calc_matrix(*args)


    def calc_matrix(self, matrix, dir:Path):
        if isinstance(matrix, Path):
            with open(matrix / utils.PORT_MATRIX_FILENAME, 'rb') as f:
                matrix = scipy.sparse.load_npz(f)
        
        denominator_data = matrix.getcol(0).toarray()
        negative_numerator = matrix.copy().tocsc()

        negative_denominator = denominator_data[0, 0] - denominator_data
        negative_denominator = 1 / negative_denominator
        negative_denominator[negative_denominator == np.inf] = 0

        negative_rate = negative_numerator.multiply(negative_denominator)
        negative_rate.eliminate_zeros()
        with open(dir / utils.NEG_RATE_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, negative_rate)

        positive_denominator = 1 / denominator_data
        positive_denominator[positive_denominator == np.inf] = 0
        positive_rate = matrix.tocsr().multiply(positive_denominator).tocsr()
        positive_rate[:, 0] = (matrix.getcol(0) / denominator_data[0, 0]).tocsr()
        positive_rate.eliminate_zeros()
            
        with open(dir / utils.POS_RATE_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, positive_rate)




    def calc(self):
        print('Model: calculating subnet matrix')
        args = []
        for subnet in (self.dir / 'subnet').iterdir():
            with open(subnet / utils.SUBNET_PORT_COUNT_FILENAME, 'r') as f:
                args.append((subnet, subnet))

        pool_cpu = multiprocessing.Pool(self.subnet_threads)
        pbar = tqdm(total=len(args), ncols=80, position=1)

        for _ in pool_cpu.imap_unordered(self.arg_wrapper_calc_matrix, args, chunksize=self.chunk_size):
            pbar.update()


    def calc_app(self):
        print('Model: calculating app matrix')
        with open(self.dir / utils.APP_MATRIX_FILENAME, 'rb') as f:
            app_matrix = scipy.sparse.load_npz(f)
        self.calc_matrix(app_matrix, self.dir)

    def arg_wrapper_extract_subnet(self, args):
        return self.extract_subnet(*args)

    def extract_subnet(self, subnet_dir:Path, service_to_index:dict):
        infos = open(subnet_dir / utils.INFO_JSON_FILENAME, 'r')

        app_num_dict = UD(name=self.shared_dict_name)

        port_col = []
        port_row = []
        app_col = []
        app_row =[]


        for line in infos.readlines():
            port_col.append(0)
            port_row.append(0)
            app_col.append(0)
            app_row.append(0)

            info = dict(json.loads(line))
            ports = info['ports']
            ports = [(x-1) * (len(service_to_index) + 1) + 1 for x in ports]
            port_row.extend([0] * len(ports))
            port_col.extend(ports)
            portServices = info['PortService']
            pS_rows = [(int(x[0])-1) * len(service_to_index) + int(service_to_index[str(x[1])]) for x in portServices]
            pS_cols = [(int(x[0])-1) * (len(service_to_index)+1) + int(service_to_index[str(x[1])]) + 1 for x in portServices]
            for i in range(len(portServices)):
                pS_row = pS_rows[i]
                
                port_row.append(0)
                port_col.append(pS_cols[i])

                port_row.extend([pS_row] * len(ports))
                port_col.extend(ports)

                port_row.append(pS_row)
                port_col.append(0)

                if i < (len(portServices) - 1):
                    port_row.extend([pS_row] * (len(portServices) - i - 1))
                    port_col.extend(pS_cols[i+1:])

                    port_row.extend(pS_rows[i+1:])
                    port_col.extend([pS_cols[i]] * (len(portServices) - i - 1))


            ip_app_feature = filter(lambda x: x is not None, map(info['other_layer'].get, self.using_app_feature))

            app_row.extend(pS_rows)
            app_col.extend([0] * len(pS_rows))
            for app_feature in ip_app_feature:
                index = app_num_dict.data[app_feature]

                app_row.extend(pS_rows)
                app_col.extend([index] * len(portServices))

                app_row.append(0)
                app_col.append(index)

        infos.close()
        port_matrix = scipy.sparse.coo_matrix((np.ones_like(port_row), (port_row, port_col)), shape=(65535*len(service_to_index)+1, 65535*(len(service_to_index)+1)+1), dtype=int)
        port_matrix.sum_duplicates()

        app_matrix = scipy.sparse.coo_matrix((np.ones_like(app_row), (app_row, app_col)), shape=(65535*len(service_to_index)+1,self.app_feature_num), dtype=int).tocsr()

        with open(subnet_dir / utils.PORT_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, port_matrix)

        with open(subnet_dir / utils.APP_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, app_matrix) 

        return





    def extract(self, using_app_feature=None):
        print('Model feature extractor: extracting from file.')
        self.using_app_feature = using_app_feature
        self.file_extractor.known_subent_extract(using_app_feature=using_app_feature)

        with open(self.dir / utils.APP_NUM_DICT_FILENAME, 'rb') as f:
            app_num_dict = pickle.load(f)

        with open(self.dir / utils.SERVICE_TO_INDEX_FILENAME, 'r') as f:
            service_to_index = json.load(f)


        self.shared_dict_name = 'app_num'
        shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

        self.app_feature_num = len(app_num_dict) + 1

        app_matrix = scipy.sparse.csr_matrix((65535*len(service_to_index)+1, self.app_feature_num), dtype=int)


        subnets = []
        subnet_dirs = []
        dir_and_port_count = {}
        for subnet_dir in (self.dir / 'subnet').iterdir():
            with open(subnet_dir / utils.SUBNET_PORT_COUNT_FILENAME, 'r') as f:
                port_count = int(f.read().strip())
                dir_and_port_count[subnet_dir] = port_count
        dir_and_port_count = sorted(dir_and_port_count.items(), key=lambda x:x[1], reverse=True)
        random.shuffle(dir_and_port_count)

        for i, item in enumerate(dir_and_port_count):
            subnet_dir, port_count = item
            subnet_dirs.append(subnet_dir)
            subnets.append((subnet_dir, service_to_index))


        if len(subnets) != 0:
            global task_queue, result_queue
            multiprocessing.set_start_method('fork', force=True)
            task_queue = multiprocessing.SimpleQueue()
            result_queue = multiprocessing.SimpleQueue()


            pool = multiprocessing.Pool(self.subnet_threads)
            print('Model feature extractor: extracting subnets.')            
            pbar = tqdm(total=len(dir_and_port_count), ncols=80, position=1)
            for _ in pool.imap_unordered(self.arg_wrapper_extract_subnet, subnets, chunksize=self.chunk_size):
                pbar.update(1)
            pool.close()
            pool.join()

            num_per_proc = math.floor(len(subnet_dirs) / self.subnet_threads)

            m_args = []
            for i in range(self.subnet_threads):
                if i != self.subnet_threads-1:
                    m_args.append((subnet_dirs[i*num_per_proc:(i+1)*num_per_proc], self.app_feature_num,65535*len(service_to_index)+1))
                else:
                    m_args.append((subnet_dirs[i*num_per_proc:], self.app_feature_num,65535*len(service_to_index)+1))
            pool = multiprocessing.Pool(self.subnet_threads)
            for m in pool.imap_unordered(args_wrapper_adder, m_args):
                app_matrix += m


        print('Model feature extractor: saving app matrix')
        with open(self.dir / utils.APP_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, app_matrix.tocoo())


        shared_app_num_dict.unlink()           
           