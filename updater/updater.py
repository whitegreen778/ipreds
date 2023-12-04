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

class Updater:
    def __init__(self, combined_dir: Path, positive_true_dir: Path, model_dir: Path, app_matrix_dir: Path,
                app_num_dir: Path, using_app_feature: list, predict_result_dir: Path,
                processes=4, groundtruth_dir=None, chunk_size=25, adders=10):
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
        self.feature_saver = FeatureSaver(self.model_dir.parent, None, subnet_threads=processes, chunk_size=chunk_size, adders=adders)
        self.adders = adders

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
            output_str = ''

            for line in input_file.readlines():
                temp = json.loads(line.strip())
                out = {}
                out['ip'] = temp.pop('ip')
                out['ports'] = []
                temp.pop('PortService')
                out['predicted_ports'] = []
                out['other_layer'] = {}
                if self.using_app_feature is not None:
                    for key in self.using_app_feature:
                        value = temp.get(key, None)
                        if value is not None:
                            out['other_layer'][key] = deepcopy(value)
                else:
                    out['other_layer'].update(temp)
                output_str += (json.dumps(out) + '\n')
                del temp, out
            output_file.write(output_str)
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
        
        positive_true_file = open(positive_true_dir / evaluator.utils.POSITIVE_TRUE_FILENAME, 'r')
        combine_file = open(combined_dir / utils.INFO_JSON_FILENAME, 'w')

        port_row = []
        port_col = []
        app_row = []
        app_col = []

        for line in positive_true_file.readlines():
            line_dict = json.loads(line)
            ip = line_dict['ip']
            for port in line_dict['ports']:
                port_row.extend([0, port])
                port_col.extend([port, 0])

                port_row.extend([port] * len(combined[ip]['ports']))
                port_col.extend(combined[ip]['ports'])

                port_row.extend(combined[ip]['ports'])
                port_col.extend([port] * len(combined[ip]['ports']))

                combined[ip]['ports'].append(port)

            ip_app_feature = filter(lambda x: x is not None, map(combined[ip]['other_layer'].get, self.using_app_feature))


            app_row.extend(line_dict['ports'])
            app_col.extend([0] * len(line_dict['ports']))
            for app_feature in ip_app_feature:
                index = app_num_dict.data[app_feature]

                app_row.append(0)
                app_col.append(index)

                app_row.extend(line_dict['ports'])
                app_col.extend([index] * len(line_dict['ports']))

        positive_true_file.close()

        port_matrix = scipy.sparse.coo_matrix((np.ones_like(port_row), (port_row, port_col)), shape=(65536,65536), dtype=int).tocsc()
        app_matrix = scipy.sparse.coo_matrix((np.ones_like(app_row), (app_row, app_col)), shape=(65536, self.app_feature_num), dtype=int).tocsc()

        with open(self.predict_result_dir / subnet / predictor.utils.PREDICT_RESULT_FILENAME, 'r') as f:
            for line in f.readlines():
                temp = json.loads(line.strip())
                combined[temp['ip']]['predicted_ports'].extend(temp['ports'])

        for ip, value in combined.items():
            value.update({'ip':ip})
            combine_file.writelines(json.dumps(value) + '\n')
        combine_file.close()

        if (model_dir / utils.PORT_MATRIX_FILENAME).exists():
            with open(model_dir / utils.PORT_MATRIX_FILENAME, 'rb') as f:
                port_matrix += scipy.sparse.load_npz(f).tocsc()
        port_matrix[0, 0] = len(combined)
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / utils.PORT_MATRIX_FILENAME, 'wb') as f:
            scipy.sparse.save_npz(f, port_matrix.tocoo())

        pos_path, pos_matrix, neg_path, neg_matrix = self.feature_saver.calc_matrix(port_matrix, model_dir, True)
        with open(pos_path, 'wb') as fo:
            scipy.sparse.save_npz(fo, pos_matrix)
        with open(neg_path, 'wb') as fo:
            scipy.sparse.save_npz(fo, neg_matrix)
        
        app_matrix[0, 0] = len(combined)

        global task_queue
        task_queue.put(app_matrix)

        return

    def update(self, init=False):
        self.init = init
        if not self.init:
            with open(self.app_num_dir, 'rb') as f:
                app_num_dict = pickle.load(f)

            self.shared_dict_name = 'app_num'
            shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

            self.app_feature_num = len(app_num_dict) + 1
            app_matrix = scipy.sparse.csc_matrix((65536, self.app_feature_num), dtype=int)

        args = []
        if not init:
            for subnet in self.positive_true_dir.iterdir():
                if subnet.is_dir():
                    args.append(subnet.name)
        else:
            for subnet in self.groundtruth_dir.iterdir():
                if subnet.is_dir():
                    args.append(subnet.name)

        if not self.init:
            global task_queue, result_queue
            multiprocessing.set_start_method('fork', force=True)
            task_queue = multiprocessing.SimpleQueue()
            result_queue = multiprocessing.SimpleQueue()

            adders_count = self.adders
            for _ in range(adders_count):
                adder_process = multiprocessing.Process(target=adder, args=(self.app_feature_num,))
                adder_process.start()

        if not self.init:
            pool = multiprocessing.Pool(processes=self.processes - adders_count)
        else:
            pool = multiprocessing.Pool(processes=self.processes)
        print('Updator: Updating subnet matrixes.')
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)

        for _ in pool.imap_unordered(self.update_subent, args, chunksize=self.chunk_size):
            pbar.update()

        if not self.init:
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

            print('Updator: saving app matrix')
            with open(self.app_matrix_dir, 'rb') as f:
                temp = app_matrix[0, 0]
                temp_matrix = scipy.sparse.load_npz(f).tocsc()
                temp_matrix.resize((65536, self.app_feature_num))
                app_matrix += temp_matrix
                app_matrix[0, 0] = temp

            with open(self.app_matrix_dir, 'wb') as f:
                scipy.sparse.save_npz(f, app_matrix.tocoo())

            print('Updator: Updating app matrix.')
            pos_path, pos_matrix, neg_path, neg_matrix = self.feature_saver.calc_matrix(app_matrix, self.app_matrix_dir.parent, False)
            with open(pos_path, 'wb') as fo:
                scipy.sparse.save_npz(fo, pos_matrix)
            with open(neg_path, 'wb') as fo:
                scipy.sparse.save_npz(fo, neg_matrix)

            shared_app_num_dict.unlink()
