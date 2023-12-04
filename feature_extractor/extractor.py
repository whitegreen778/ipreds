import os
from pathlib import Path
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
import pickle
from tqdm import tqdm
import numpy as np
import scipy.sparse
import json
from UltraDict import UltraDict as UD
from tqdm import tqdm
import pickle
import os
import random
import feature_extractor.utils as utils

import warnings
warnings.filterwarnings("ignore")

def adder(app_feature_num: int):
    global task_queue, result_queue
    
    matrix = scipy.sparse.csc_matrix((65536, app_feature_num), dtype=int)
    while True:
        item = task_queue.get()
        if item is not None:
            matrix += item
        else:
            result_queue.put(matrix)
            result_queue.put(None)
            break


class ExtractorOfFile(ABC):

    def __init__(self, dir, processes=(os.cpu_count() - 2)):
        self.dir = Path(dir)
        (self.dir / 'subnet').mkdir(exist_ok=True, parents=True)
        self.processes = processes


    @abstractmethod
    def subnet_extract(self, subnet_dir:Path):
        pass

    def known_subent_extract(self, using_app_feature:set, need_port_count=True, need_app_num_dict=True):
        self.using_app_feature = using_app_feature
        self.need_port_count = need_port_count
        self.need_app_num_dict = need_app_num_dict

        if need_app_num_dict or need_port_count:
            args = []
            for subnet_dir in (self.dir / 'subnet').iterdir():
                if subnet_dir.is_dir():
                    args.append(subnet_dir)
            
            pbar = tqdm(total=len(args), ascii=True)
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
                subnet_threads=4,  
                chunk_size=25, adders=10,
                matrix_save_buffer_size=100):
        self.dir = Path(dir)
        self.file_extractor = file_extractor
        (self.dir / 'subnet').mkdir(exist_ok=True, parents=True)
        self.subnet_threads = subnet_threads
        self.chunk_size = chunk_size
        self.adders = adders
        self.matrix_save_buffer_size = matrix_save_buffer_size



    def arg_wrapper_calc_matrix(self, args):
        return self.calc_matrix(*args)


    def calc_matrix(self, matrix, dir:Path, first_row_negative=False):
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

        positive_denominator = 1 / denominator_data
        positive_denominator[positive_denominator == np.inf] = 0
        positive_rate = matrix.tocsr().multiply(positive_denominator)
        positive_rate.eliminate_zeros()
            
        return dir / utils.POS_RATE_FILENAME, positive_rate, dir / utils.NEG_RATE_FILENAME, negative_rate




    def calc(self):
        print('Model: calculating subnet matrix')
        args = []
        for subnet in (self.dir / 'subnet').iterdir():
            with open(subnet / utils.SUBNET_PORT_COUNT_FILENAME, 'r') as f:
                args.append((subnet, subnet, True))

        pool_cpu = multiprocessing.Pool(self.subnet_threads)
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)
        temp_path_and_matrix = []

        for result in pool_cpu.imap_unordered(self.arg_wrapper_calc_matrix, args, chunksize=25):
            temp_path_and_matrix.append(result)
            if len(temp_path_and_matrix) == self.matrix_save_buffer_size:
                for item in temp_path_and_matrix:
                    pos_path, pos_matrix, neg_path, neg_matrix = item
                    with open(pos_path, 'wb') as fo:
                        scipy.sparse.save_npz(fo, pos_matrix)
                    with open(neg_path, 'wb') as fo:
                        scipy.sparse.save_npz(fo, neg_matrix)
                temp_path_and_matrix = []
            pbar.update()
        for item in temp_path_and_matrix:
            pos_path, pos_matrix, neg_path, neg_matrix = item
            with open(pos_path, 'wb') as fo:
                scipy.sparse.save_npz(fo, pos_matrix)
            with open(neg_path, 'wb') as fo:
                scipy.sparse.save_npz(fo, neg_matrix)
        temp_path_and_matrix = []


    def calc_app(self):
        print('Model: calculating app matrix')
        with open(self.dir / utils.APP_MATRIX_FILENAME, 'rb') as f:
            app_matrix = scipy.sparse.load_npz(f)
        pos_path, pos_matrix, neg_path, neg_matrix = self.calc_matrix(app_matrix, self.dir, False)
        with open(pos_path, 'wb') as fo:
            scipy.sparse.save_npz(fo, pos_matrix)
        with open(neg_path, 'wb') as fo:
            scipy.sparse.save_npz(fo, neg_matrix)


    def arg_wrapper_extract_subnet(self, args):
        return self.extract_subnet(*args)

    def extract_subnet(self, subnet_dir):
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
            info_ports_len = len(info['ports'])
            for i in range(info_ports_len):
                port = info['ports'][i]
                port_row.extend([port, 0])
                port_col.extend([0, port])

                if i < (info_ports_len - 1):
                    port_row.extend([port] * (info_ports_len - i - 1))
                    port_col.extend(info['ports'][i+1:])

                    port_row.extend(info['ports'][i+1:])
                    port_col.extend([port] * (info_ports_len - i - 1))


            ip_app_feature = filter(lambda x: x is not None, map(info['other_layer'].get, self.using_app_feature))

            app_col.extend([0] * info_ports_len)
            app_row.extend(info['ports'])
            for app_feature in ip_app_feature:
                index = app_num_dict.data[app_feature]

                app_row.append(0)
                app_col.append(index)

                app_row.extend(info['ports'])
                app_col.extend([index] * info_ports_len)

        port_matrix = scipy.sparse.coo_matrix((np.ones_like(port_row), (port_row, port_col)), shape=(65536, 65536))
        port_matrix.sum_duplicates()

        app_matrix = scipy.sparse.coo_matrix((np.ones_like(app_row), (app_row, app_col)), shape=(65536,self.app_feature_num)).tocsr()


        global task_queue
        task_queue.put(app_matrix)

        return subnet_dir / utils.PORT_MATRIX_FILENAME, port_matrix





    def extract(self, using_app_feature=None):
        print('Model feature extractor: extracting from file.')
        self.subnet_known = True
        self.using_app_feature = using_app_feature
        self.file_extractor.known_subent_extract(using_app_feature=using_app_feature)

        with open(self.dir / utils.APP_NUM_DICT_FILENAME, 'rb') as f:
            app_num_dict = pickle.load(f)

        self.shared_dict_name = 'app_num'
        shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

        self.app_feature_num = len(app_num_dict) + 1

        app_matrix = scipy.sparse.csr_matrix((65536, self.app_feature_num), dtype=int)


        subnets = []
        dir_and_port_count = {}
        for subnet_dir in (self.dir / 'subnet').iterdir():
            with open(subnet_dir / utils.SUBNET_PORT_COUNT_FILENAME, 'r') as f:
                port_count = int(f.read().strip())
                dir_and_port_count[subnet_dir] = port_count
        dir_and_port_count = sorted(dir_and_port_count.items(), key=lambda x:x[1], reverse=True)
        random.shuffle(dir_and_port_count)

        for i, item in enumerate(dir_and_port_count):
            subnet_dir, port_count = item
            subnets.append(subnet_dir)


        if len(subnets) != 0:
            global task_queue, result_queue
            multiprocessing.set_start_method('fork', force=True)
            task_queue = multiprocessing.SimpleQueue()
            result_queue = multiprocessing.SimpleQueue()

            adders_count = self.adders
            for _ in range(adders_count):
                adder_process = multiprocessing.Process(target=adder, args=(self.app_feature_num,))
                adder_process.start()


            pool = multiprocessing.Pool(self.subnet_threads - adders_count)
            print('Model feature extractor: extracting subnets.')            
            pbar = tqdm(total=len(dir_and_port_count), ncols=80, position=1, ascii=True)
            temp_port_matrix_list = []
            for path_and_matrix in pool.imap_unordered(self.extract_subnet, subnets, chunksize=self.chunk_size):
                temp_port_matrix_list.append(path_and_matrix)
                if len(temp_port_matrix_list) == self.matrix_save_buffer_size:
                    for item in temp_port_matrix_list:
                        path, matrix = item
                        with open(path, 'wb') as fo:
                            scipy.sparse.save_npz(fo, matrix)
                    temp_port_matrix_list = []
                pbar.update(1)

            for item in temp_port_matrix_list:
                path, matrix = item
                with open(path, 'wb') as fo:
                    scipy.sparse.save_npz(fo, matrix)
            temp_port_matrix_list = []

            for _ in range(adders_count):
                task_queue.put(None)
            
            none_count = 0
            while True:
                item = result_queue.get()
                if item is None:
                    none_count += 1
                    if none_count == adders_count:
                        break
                else:
                    app_matrix += item


        print('Model feature extractor: saving app matrix')
        with open(self.dir / utils.APP_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, app_matrix.tocoo())


        shared_app_num_dict.unlink()           
           
