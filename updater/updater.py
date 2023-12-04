from pathlib import Path
import multiprocessing
from tqdm import tqdm
import json
import feature_extractor.utils as utils
from feature_extractor.extractor import FeatureSaver
import pickle
from UltraDict import UltraDict as UD
import scipy.sparse
from copy import deepcopy
import evaluator.utils
import predictor.utils
import numpy as np
import math

def adder(app_feature_num: int, row:int):
    global task_queue, result_queue
    
    matrix = scipy.sparse.csc_matrix((row, app_feature_num), dtype=int)
    while True:
        item = task_queue.get()
        if item is not None:
            matrix += item
        else:
            result_queue.put(matrix)
            result_queue.put(None)
            break

def args_wrapper_adder(args):
    return add_app_matrix(*args)

def add_app_matrix(dirs:Path, app_feature_num:int, row:int):
    matrix = scipy.sparse.csc_matrix((row, app_feature_num), dtype=int)
    for subnet in dirs:
        matrix += scipy.sparse.load_npz(subnet / utils.APP_MATRIX_FILENAME)
    return matrix

class Updater:
    def __init__(self, combined_dir: Path, positive_true_dir: Path, model_dir: Path, app_matrix_dir: Path,
                app_num_dir: Path, using_app_feature: list, predict_result_dir: Path,
                processes=4, groundtruth_dir=None, chunk_size=25):
        self.combined_dir = combined_dir
        self.positive_true_dir = positive_true_dir
        self.model_dir = model_dir
        self.processes = processes
        self.groundtruth_dir = groundtruth_dir
        self.app_matrix_dir = app_matrix_dir
        self.app_num_dir = app_num_dir
        self.predict_result_dir = predict_result_dir
        self.using_app_feature = using_app_feature
        self.chunk_size = chunk_size
        self.feature_saver = FeatureSaver(self.model_dir.parent, None, subnet_threads=processes, chunk_size=chunk_size)

    def update_subent(self, subnet: str):
        combined_dir = self.combined_dir / subnet
        positive_true_dir = self.positive_true_dir / subnet
        model_dir = self.model_dir / subnet

        if self.init:
            combined_dir.mkdir(exist_ok=True, parents=True)
            if self.groundtruth_dir is None:
                raise Exception('Need groundtruth')

            groundtruth_dir = self.groundtruth_dir / subnet

            input_file = open(groundtruth_dir / utils.INFO_JSON_FILENAME, 'r')
            output_file = open(combined_dir / utils.INFO_JSON_FILENAME, 'w')

            out_str = ''
            for line in input_file.readlines():
                temp = json.loads(line.strip())
                out = {}
                out['ip'] = temp.pop('ip')
                out['ports'] = temp.pop('ports')
                out['PortService'] = []
                out['predicted_services'] = {}
                out['other_layer'] = {}
                if self.using_app_feature is not None:
                    for key in self.using_app_feature:
                        value = temp.get(key, None)
                        if value is not None:
                            out['other_layer'][key] = deepcopy(value)
                else:
                    out['other_layer'].update(temp)
                out_str += json.dumps(out) + '\n'
                del temp, out
            output_file.write(out_str)
            input_file.close()
            output_file.close()
            return

        app_num_dict = UD(name=self.shared_dict_name)

        combined = {}
        with open(combined_dir / utils.INFO_JSON_FILENAME, 'r') as f:
            for line in f.readlines():
                temp = json.loads(line.strip())
                ip = temp.pop('ip')
                combined[ip] = temp
                del temp
        
        positive_true_file = open(positive_true_dir / evaluator.utils.POSITIVE_TRUE_FILENAME, 'r')
        combine_file = open(combined_dir / utils.INFO_JSON_FILENAME, 'w')

        global service_to_index, index_to_service

        service_row = []
        service_col = []
        app_row = []
        app_col = []

        for line in positive_true_file.readlines():
            line_dict = json.loads(line)
            ip = line_dict['ip']
            ports = combined[ip]['ports']
            p_col = [(x-1) * (len(service_to_index) + 1) + 1 for x in ports]
            ori_pSes = combined[ip]['PortService']
            ori_pS_rows = [(int(x[0])-1) * len(service_to_index) + int(service_to_index[str(x[1])]) for x in ori_pSes]
            ori_pS_cols = [(int(x[0])-1) * (len(service_to_index)+1) + int(service_to_index[str(x[1])]) + 1 for x in ori_pSes]
            
            ip_app_feature = filter(lambda x: x is not None, map(combined[ip]['other_layer'].get, self.using_app_feature))
            
            for pS in line_dict['PortService']:
                row = (int(pS[0])-1) * len(service_to_index) + int(service_to_index[str(pS[1])])
                col = (int(pS[0])-1) * (len(service_to_index)+1) + int(service_to_index[str(pS[1])])

                service_row.append(row)
                service_col.append(0)

                service_row.append(0)
                service_col.append(col)

                service_row.extend([row] * len(p_col))
                service_col.extend(p_col)

                service_row.extend([row] * len(ori_pS_cols))
                service_col.extend(ori_pS_cols)

                service_row.extend(ori_pS_rows)
                service_col.extend([col] * len(ori_pS_rows))


                
                app_row.append(row)
                app_col.append(0)
                for app_feature in ip_app_feature:
                    appindex = app_num_dict.data[app_feature]
                    app_row.append(0)
                    app_col.append(appindex)

                    app_row.append(row)
                    app_col.append(appindex)
                
                combined[ip]['PortService'].append(pS)

        positive_true_file.close()

        port_matrix = scipy.sparse.coo_matrix((np.ones_like(service_row), (service_row, service_col)), shape=(65535*len(service_to_index)+1, 65535*(len(service_to_index)+1)+1), dtype=int).tocsc()
        app_matrix = scipy.sparse.coo_matrix((np.ones_like(app_row), (app_row, app_col)), shape=(65535*len(service_to_index)+1,self.app_feature_num), dtype=int).tocsc()

        with open(self.predict_result_dir / subnet / predictor.utils.PREDICT_RESULT_FILENAME, 'r') as f:
            for line in f.readlines():
                temp = json.loads(line.strip())
                for p, services in temp['PortService'].items():
                    if p not in combined[temp['ip']]['predicted_services']:
                        combined[temp['ip']]['predicted_services'][p] = services
                    else:
                        combined[temp['ip']]['predicted_services'][p].extend(services)

        combine_output_str = ''
        for ip, value in combined.items():
            value.update({'ip':ip})
            combine_output_str += json.dumps(value) + '\n'
        combine_file.write(combine_output_str)
        combine_file.close()

        if (model_dir / utils.PORT_MATRIX_FILENAME).exists():
            with open(model_dir / utils.PORT_MATRIX_FILENAME, 'rb') as f:
                port_matrix += scipy.sparse.load_npz(f).tocsc()
        port_matrix[0, 0] = len(combined)
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / utils.PORT_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, port_matrix.tocoo())

        self.feature_saver.calc_matrix(port_matrix, model_dir)
        
        app_matrix[0, 0] = len(combined)

        with open(model_dir / utils.APP_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, app_matrix)

        return

    def update(self, init=False):
        self.init = init
        if not self.init:
            global service_to_index, index_to_service
            with open(self.model_dir.parent / utils.SERVICE_TO_INDEX_FILENAME, 'r') as f:
                service_to_index = json.load(f)
            with open(self.model_dir.parent / utils.INDEX_TO_SERVICE_FILENAME, 'r') as f:
                temp = json.load(f)
                index_to_service = {}
                for k, v in temp.items():
                    index_to_service[int(k)] = v
            with open(self.app_num_dir, 'rb') as f:
                app_num_dict = pickle.load(f)

            self.shared_dict_name = 'app_num'
            shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

            self.app_feature_num = len(app_num_dict) + 1
            app_matrix = scipy.sparse.csc_matrix((65535*len(service_to_index)+1, self.app_feature_num), dtype=int)


        args = []
        subnet_dirs = []
        if not init:
            for subnet in self.positive_true_dir.iterdir():
                if subnet.is_dir():
                    args.append(subnet.name)
                    subnet_dirs.append(self.model_dir / subnet.name)
        else:
            for subnet in self.groundtruth_dir.iterdir():
                if subnet.is_dir():
                    args.append(subnet.name)

        if not self.init:
            global task_queue, result_queue
            multiprocessing.set_start_method('fork', force=True)
            task_queue = multiprocessing.SimpleQueue()
            result_queue = multiprocessing.SimpleQueue()

        pool = multiprocessing.Pool(processes=self.processes)
        print('Updator: Updating subnet matrixes.')
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)

        for _ in pool.imap_unordered(self.update_subent, args, chunksize=self.chunk_size):
            pbar.update()
        pool.close()
        pool.join()

        if not self.init:
            num_per_proc = math.floor(len(subnet_dirs) / self.processes)

            m_args = []
            for i in range(self.processes):
                if i != self.processes-1:
                    m_args.append((subnet_dirs[i*num_per_proc:(i+1)*num_per_proc], self.app_feature_num,65535*len(service_to_index)+1))
                else:
                    m_args.append((subnet_dirs[i*num_per_proc:], self.app_feature_num,65535*len(service_to_index)+1))
            pool = multiprocessing.Pool(self.processes)
            for m in pool.imap_unordered(args_wrapper_adder, m_args):
                app_matrix += m
            

            print('Updator: saving app matrix')
            with open(self.app_matrix_dir, 'rb') as f:
                temp = app_matrix[0, 0]
                temp_matrix = scipy.sparse.load_npz(f).tocsc()
                temp_matrix.resize((65535*len(service_to_index)+1, self.app_feature_num))
                app_matrix += temp_matrix
                app_matrix[0, 0] = temp

            with open(self.app_matrix_dir, 'wb') as f:
                scipy.sparse.save_npz(f, app_matrix.tocoo())

            print('Updator: Updating app matrix.')
            self.feature_saver.calc_matrix(app_matrix, self.app_matrix_dir.parent)

            shared_app_num_dict.unlink()
