from pathlib import Path
import multiprocessing
import pickle
import numpy as np
import scipy.sparse
import json
from UltraDict import UltraDict as UD
from tqdm import tqdm
import pickle
from predictor.extractor import CensysExtractorForTest
import predictor.utils
from copy import deepcopy
import feature_extractor.utils
import predictor.utils
import time
import numba
from numba import njit


def adder(result_queue: multiprocessing.SimpleQueue):
    global task_queue
    
    all_scanned_time = np.zeros(65536, dtype=int)
    new_app_set = set()
    while True:
        item = task_queue.get()
        if item is not None:
            all_scanned_time += item[1]
            new_app_set.update(item[2])
        else:
            result_queue.put(all_scanned_time)
            result_queue.put(new_app_set)
            result_queue.put(None)
            break

@njit(fastmath=True, cache=False)
def calc(port_indexs:list, app_indexs:list, app_pos_first_col:np.array,
        app_pos:np.array, port_positive_first_col:np.array, port_pos:np.array,
        predicted_ports:list, predict_num:int, uncertainty:np.array,
        app_neg_first_col:np.array, app_neg:np.array, 
        port_negative_first_col:np.array, port_neg:np.array):
        subnet_scanned_times = np.zeros(65536, dtype=numba.int64)
        results = np.zeros((len(app_indexs), predict_num), dtype=numba.int64)

        for i in range(len(app_indexs)):
            if i != len(app_indexs) - 1:
                app_pos_score = app_pos_first_col * 1
                app_neg_score = app_neg_first_col * 1
                for index in app_indexs[i]:
                    app_pos_score *= app_pos[:, index]
                    app_neg_score *= app_neg[:, index]

                port_pos_score = port_positive_first_col * 1
                port_neg_score = port_negative_first_col * 1
                for index in port_indexs[i]:
                    port_pos_score *= port_pos[:, index]
                    port_neg_score *= port_neg[:, index]

                app_score = 1 / (1 + app_neg_score / app_pos_score)
                app_score[app_score == np.nan] = 0
                port_score = 1 / (1 + port_neg_score / port_pos_score)
                port_score[port_score == np.nan] = 0
                score = app_score * 0.1 + port_score + uncertainty
                score[predicted_ports[i]] = 0
                score = score[1:]
                index = np.where(score != 0)[0]
                score = score[index]
                result = index[np.argsort(score)][::-1][:predict_num] + 1
                results[i, :result.shape[0]] = result
                subnet_scanned_times[result] += 1
        return results, subnet_scanned_times


class Predictor:
    def __init__(self, input_dir:Path, data_dir:Path, output_dir:Path,
                        processes=4, uncertainty_weigh=0.1,
                        extractor_processes=4, 
                        predict_port_num=100, chunk_size=25, adders=10,
                        matrix_save_buffer_size=100):
        self.input_dir = input_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processes = processes
        self.predict_port_num = predict_port_num
        self.uncertainty_weigh = uncertainty_weigh
        self.chunk_size = chunk_size
        self.adders = adders
        self.file_extractor = CensysExtractorForTest(self.input_dir, extractor_processes)
        self.matrix_save_buffer_size = matrix_save_buffer_size
        
        row = []
        col = []
        data = []
        for i in range(65536):
            row.append(i)
            col.append(0)
            data.append(1)
            if i != 0:
                row.append(i)
                col.append(i)
                data.append(-1)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        self.minus_matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=(65536, 65536))
        del col
        del data
        del row

    
    def arg_wrapper_pred_subnet(self, args):
        return self.predict_subnet(*args)


    def predict_subnet(self, subnet_dir:Path):
        time1 = time.time()

        global task_queue, app_positive_rate, app_negative_rate, app_denominator, app_positive_count

        
        subnet = subnet_dir.name
        output_dir = self.output_dir / 'subnet' / subnet
        output_dir.mkdir(exist_ok=True, parents=True)

        invalid_feat = set()

        old_subnet_scanned_times = np.zeros(65536, dtype=np.float32)
        data_dir = self.data_dir / 'subnet' / subnet
        if data_dir.exists():
            with open(data_dir / 'positive_rate.npz', 'rb') as f:
                port_positive = scipy.sparse.load_npz(f)
            with open(data_dir / 'port_matrix.npz', 'rb') as f:
                port_matrix = scipy.sparse.load_npz(f)
            with open(data_dir / 'negative_rate.npz', 'rb') as f:
                port_negative = scipy.sparse.load_npz(f)

            temp_col = port_positive.getrow(0).transpose()
            port_positive = port_positive.tocsc()
            port_positive_first_col = temp_col.toarray().reshape(-1).astype(np.float32)
            port_positive_first_col[0] = 0

            port_negative = port_negative.tocsc()
            port_negative_first_col = self.minus_matrix.tocsc().dot(temp_col).toarray().reshape(-1)
            port_negative_first_col[0] = 0
            port_denominator = 1 / (self.minus_matrix.tocsc().dot(port_matrix.getcol(0)).toarray())
            port_denominator[np.isinf(port_denominator)] = 0
            port_matrix = port_matrix.tocsr()

            with open(self.input_dir / 'subnet' / subnet / 'port_set.pkl', 'rb') as f:
                port_set = pickle.load(f)
            
            port_pos = np.zeros((65536, len(port_set)), order='F', dtype=np.float32)
            port_neg = np.zeros_like(port_pos)
            port_map = {}
            for index, port in enumerate(port_set):
                if port_matrix[0, port] == 0:
                    invalid_feat.add(port)
                    continue
                port_pos[:, index] = port_positive.getcol(port).toarray().reshape(-1).astype(np.float32)
                port_neg[:, index] = (port_matrix[0, port] * port_denominator - port_negative.getcol(port).toarray()).reshape(-1).astype(np.float32)
                port_map[port] = index

            del temp_col, port_positive, port_negative, port_matrix, port_denominator

            if (data_dir / predictor.utils.SCANNED_TIMES_FILENAME).exists():
                old_subnet_scanned_times = scipy.sparse.load_npz(data_dir / predictor.utils.SCANNED_TIMES_FILENAME).toarray().reshape(-1).astype(np.float32)

        app_num_dict = UD(name=self.shared_dict_name)
        app_pos_first_col = app_positive_rate.getcol(0).toarray().reshape(-1).astype(np.float32)
        app_neg_first_col = app_negative_rate.getcol(0).toarray().reshape(-1).astype(np.float32)

        with open(self.input_dir / 'subnet' / subnet / 'app_set.pkl', 'rb') as f:
            app_set = pickle.load(f)

        app_pos = np.zeros((65536, len(app_set)), order='F', dtype=np.float32)
        app_neg = np.zeros_like(app_pos)
        app_map = {}
        new_app_set = set()
        for index, app in enumerate(app_set):
            ori_index = app_num_dict.data.get(app, None)
            if ori_index is not None:
                app_pos[:, index] = app_positive_rate.getcol(ori_index).toarray().reshape(-1).astype(np.float32)
                app_neg[:, index] = (app_positive_count[ori_index] * app_denominator - app_negative_rate.getcol(ori_index).toarray()).reshape(-1).astype(np.float32)
                app_map[app] = index
            else:
                new_app_set.add(app)


        with open(subnet_dir / 'infos.json', 'r') as f:
            output_dir = self.output_dir / 'subnet' / subnet
            output_dir.mkdir(exist_ok=True, parents=True)
            lines = f.readlines()
        
        output_string = ''
        ips = []
        port_indexes = []
        app_indexes = []
        predicted_ports = []

        for i in range(len(lines)):
            line = lines[i]
            info = dict(json.loads(line))
                        
            ips.append(info['ip'])
            ports = list(set(info['ports']) - invalid_feat)
            port_indexes.append(np.array(list(map(port_map.get, ports)), dtype=int))
            app_indexes.append(np.array(list(filter(lambda x: x is not None, map(app_map.get, info['other_layer'].values()))), dtype=int))
            predicted_ports.append(np.array(info['predicted_ports'], dtype=int))

        port_indexes.append(np.zeros(0, dtype=int))
        app_indexes.append(np.zeros(0, dtype=int))
        predicted_ports.append(np.zeros(0, dtype=int))
        total_scanned_times = np.sum(old_subnet_scanned_times, dtype=np.float32)
        uncertainty = np.zeros(65536, dtype=np.float32)
        if total_scanned_times >= 1 and self.uncertainty_weigh != 0:
            uncertainty = np.sqrt(2 * np.log(total_scanned_times) / (old_subnet_scanned_times + 1))
            uncertainty *= np.array(self.uncertainty_weigh, dtype=np.float32)


        results, new_subnet_scanned_times = calc(port_indexes, app_indexes, app_pos_first_col,
                                        app_pos, port_positive_first_col, port_pos,
                                        predicted_ports, self.predict_port_num, uncertainty,
                                        app_neg_first_col, app_neg, port_negative_first_col, 
                                        port_neg)

        time2 = time.time()

        for i in range(len(ips)):
            output_string += (json.dumps({'ip':ips[i], 'ports':results[i, :].tolist()}) + '\n')

        with open(output_dir / predictor.utils.PREDICT_USING_TIME_FILENAME, 'w') as fo:
            fo.write(str(time2-time1)+'\n')
        if data_dir.exists():
            subnet_scanned_times = scipy.sparse.csr_matrix(old_subnet_scanned_times.astype(np.int64) + new_subnet_scanned_times)
            scipy.sparse.save_npz(data_dir / predictor.utils.SCANNED_TIMES_FILENAME, subnet_scanned_times)

        task_queue.put((True, new_subnet_scanned_times, new_app_set))
        return (True, output_dir / predictor.utils.PREDICT_RESULT_FILENAME, output_string)
      
                



    def predict(self, using_app_feature=None):
        if not (self.output_dir / 'epoch.txt').exists():
            with open((self.output_dir / 'epoch.txt'), 'w') as f:
                f.write(str(1))
            self.epoch = 1
        else:
            with open((self.output_dir / 'epoch.txt'), 'r') as f:
                self.epoch = int(f.read().strip())


        self.using_app_feature = using_app_feature
        print('Predictor: extracting from file')
        self.file_extractor.known_subent_extract(using_app_feature=using_app_feature)


        global task_queue, app_positive_rate, app_negative_rate, app_denominator, app_positive_count

        with open(self.data_dir / feature_extractor.utils.APP_NUM_DICT_FILENAME, 'rb') as f:
            app_num_dict = pickle.load(f)

        self.shared_dict_name = 'app_num'
        shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

        self.app_feature_num = len(app_num_dict) + 1

        with open(self.data_dir / 'app_matrix.npz', 'rb') as f:
            app_matrix = scipy.sparse.load_npz(f)
        with open(self.data_dir / 'positive_rate.npz', 'rb') as f:
            app_positive_rate = scipy.sparse.load_npz(f).tocsc()
        with open(self.data_dir / 'negative_rate.npz', 'rb') as f:
            app_negative_rate = scipy.sparse.load_npz(f).tocsc()
        
        temp_col = app_matrix.getcol(0)
        app_positive_rate[:, 0] = temp_col / temp_col[0, 0]

        app_negative_rate[:, 0] = self.minus_matrix.tocsc().dot(app_positive_rate.getcol(0))
        app_denominator = 1 / self.minus_matrix.tocsc().dot(app_matrix.getcol(0)).toarray()
        app_denominator[np.isinf(app_denominator)] = 0
        app_positive_count = app_matrix.getrow(0).toarray().reshape(-1)

        self.all_scanned_times = np.zeros(65536, dtype=int)
        if (self.data_dir / predictor.utils.SCANNED_TIMES_FILENAME).exists():
            self.all_scanned_times = np.load(self.data_dir / predictor.utils.SCANNED_TIMES_FILENAME)
        all_scanned_times_in_func = deepcopy(self.all_scanned_times)

        new_app_set = set()
        
        args = []
        for subnet_dir in (self.input_dir / 'subnet').iterdir():
            with open(subnet_dir / predictor.utils.LINE_COUNT_FILENAME, 'r') as f:    
                args.append((subnet_dir, int(f.read().strip())))
        args = sorted(args, key=lambda x:x[1], reverse=True)
        args = [x[0] for x in args]


        multiprocessing.set_start_method('fork', force=True)
        task_queue = multiprocessing.SimpleQueue()
        result_queue = multiprocessing.SimpleQueue()
        adders_count = self.adders
        for _ in range(adders_count):
            adder_process = multiprocessing.Process(target=adder, args=(result_queue,))
            adder_process.start()

        pool = multiprocessing.Pool(self.processes-adders_count)
        print('Predictor: Predicting with different subnets')
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)
        temp_path_and_result = []
        pool = multiprocessing.Pool(self.processes-adders_count)
        for result in pool.imap_unordered(self.predict_subnet, args, chunksize=self.chunk_size):
            temp_path_and_result.append(result)

            if len(temp_path_and_result) == self.matrix_save_buffer_size:
                for item in temp_path_and_result:
                    _, path, predict_result = item
                    with open(path, 'w') as fo:
                        fo.write(predict_result)
                temp_path_and_result = []
            pbar.update()
        pool.close()
        pool.join()
        for item in temp_path_and_result:
            _, path, predict_result = item
            with open(path, 'w') as fo:
                fo.write(predict_result)
        temp_path_and_result = []

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
                if isinstance(item, set):
                    new_app_set.update(item)
                else:
                    all_scanned_times_in_func += item
                

        np.save(self.data_dir / predictor.utils.SCANNED_TIMES_FILENAME, all_scanned_times_in_func)
        with open((self.output_dir / 'epoch.txt'), 'w') as f:
            f.write(str(self.epoch + 1))

        print('Predictor: Updating app_num_dict')
        for app in new_app_set:
            app_num_dict[app] = len(app_num_dict) + 1

        with open(self.data_dir / feature_extractor.utils.APP_NUM_DICT_FILENAME, 'wb') as f:
            pickle.dump(app_num_dict, f)
