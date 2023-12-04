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
from numba import njit
from numba.typed import Dict
from numba.core import types


def adder(result_queue: multiprocessing.SimpleQueue, row:int):
    global task_queue
    
    all_scanned_time = np.zeros(row, dtype=int)
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

@njit(fastmath=True, cache=True)
def calc(all_row_index:list, all_feature_cols:list, all_app_cols:list,
        all_predicted_service:list, app_pos:np.ndarray, port_pos:np.ndarray,
        app_neg:np.ndarray, port_neg:np.ndarray,
        port_first_col:np.ndarray, app_first_col:np.ndarray,
        port_neg_first_col:np.ndarray, app_neg_first_col:np.ndarray,
        services_count:int, predict_num:int, uncertainty:np.ndarray):
        results = []

        for i in range(len(all_row_index)):
            if i != len(all_row_index) - 1:
                candidate_row_index = all_row_index[i]
                per_ip_result = []
                for j in range(candidate_row_index.shape[0]):
                    start_row = candidate_row_index[j] * services_count
                    end_row = start_row + services_count
                    
                    app_pos_score = app_first_col[start_row:end_row] * 1
                    app_neg_score = app_neg_first_col[start_row:end_row] * 1
                    
                    app_cols = all_app_cols[i]
                    for col_i in range(app_cols.shape[0]):
                        app_pos_score *= app_pos[start_row:end_row, app_cols[col_i]]
                        app_neg_score *= app_neg[start_row:end_row, app_cols[col_i]]

                    port_pos_score = port_first_col[start_row:end_row] * 1
                    port_neg_score = port_neg_first_col[start_row:end_row] * 1

                    port_cols = all_feature_cols[i]
                    for col_i in range(port_cols.shape[0]):
                        port_pos_score *= port_pos[start_row:end_row, port_cols[col_i]]
                        port_neg_score *= port_neg[start_row:end_row, port_cols[col_i]]

                    port_score = 1 / (1 + port_neg_score / port_pos_score)
                    port_score[np.isnan(port_score)] = 0

                    app_score = 1 / (1 + app_neg_score / app_pos_score)
                    app_score[np.isnan(app_score)] = 0

                    score = app_score + port_score + uncertainty[start_row:end_row]
                    if candidate_row_index[j] in all_predicted_service[i]:
                        score[all_predicted_service[i][(candidate_row_index[j]-1)]] = 0
                    per_ip_result.append(np.argsort(score)[::-1][:predict_num] + 1)
                results.append(per_ip_result)
        return results

class Predictor:
    def __init__(self, input_dir:Path, data_dir:Path, output_dir:Path,
                        processes=4, uncertainty_weigh=0.1,
                        extractor_processes=4, adders=1,
                        predict_port_num=100, chunk_size=25):
        self.input_dir = input_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processes = processes
        self.predict_port_num = predict_port_num
        self.uncertainty_weigh = uncertainty_weigh
        self.chunk_size = chunk_size
        self.adders = adders
        self.file_extractor = CensysExtractorForTest(self.input_dir, extractor_processes)
        

    
    def arg_wrapper_pred_subnet(self, args):
        return self.predict_subnet(*args)


    def predict_subnet(self, subnet_dir:Path):
        time1 = time.time()
        global task_queue, app_positive_rate, service_to_index, index_to_service, app_negative_rate, app_denominator, app_positive_count

        
        subnet = subnet_dir.name
        output_dir = self.output_dir / 'subnet' / subnet
        output_dir.mkdir(exist_ok=True, parents=True)

        old_subnet_scanned_times = np.zeros(65535*len(service_to_index)+1, dtype=int)
        data_dir = self.data_dir / 'subnet' / subnet

        invalid_feat = set()

        if data_dir.exists():
            with open(data_dir / 'positive_rate.npz', 'rb') as f:
                port_positive = scipy.sparse.load_npz(f).tocsc()
            if True:
                with open(data_dir / 'port_matrix.npz', 'rb') as f:
                    port_matrix = scipy.sparse.load_npz(f)
                with open(data_dir / 'negative_rate.npz', 'rb') as f:
                    port_negative = scipy.sparse.load_npz(f)

            port_positive = port_positive.tocsc()
            port_positive_first_col = port_positive.getcol(0).toarray().reshape(-1).astype(np.float32)
            if True:
                port_negative = port_negative.tocsc()

                m_row = list(range(port_positive.shape[0]))
                m_col = [0] * len(m_row)
                m_data = [1] * len(m_row)
                m_row.extend(list(range(1, port_positive.shape[0])))
                m_col.extend(list(range(1, port_positive.shape[0])))
                m_data.extend([-1] * (port_positive.shape[0] - 1))
                port_negative_first_col = scipy.sparse.coo_matrix(
                    (m_data, (m_row, m_col)), shape=(port_positive.shape[0], port_positive.shape[0])
                    ).dot(port_positive.getcol(0)).toarray().reshape(-1).astype(np.float32)
                port_negative_first_col[0] = 0

                port_denominator = 1 / (
                        scipy.sparse.coo_matrix(
                            (m_data, (m_row, m_col)), shape=(port_positive.shape[0], port_positive.shape[0])
                        ).dot(port_matrix.getcol(0)).toarray().reshape(-1).astype(np.float32)
                    )
                port_matrix = port_matrix.tocsr()
                port_matrix_first_row = port_matrix[0, :].toarray().astype(np.float32)

            with open(self.input_dir / 'subnet' / subnet / predictor.utils.SUBNET_SERVICE_SET_FILENAME, 'rb') as f:
                service_set = pickle.load(f)

            with open(self.input_dir / 'subnet' / subnet / predictor.utils.SUBNET_NEED_PRED_PORT_FILENAME, 'rb') as f:
                need_pred_port_list = pickle.load(f)

            start_row = [(x-1) * len(service_to_index)+1 for x in need_pred_port_list]
            end_row = [x * len(service_to_index)+1 for x in need_pred_port_list]

            port_to_index_dict = {}
            for i in range(len(need_pred_port_list)):
                port_to_index_dict[need_pred_port_list[i]] = i

            
            port_pos = np.zeros((len(need_pred_port_list)*len(service_to_index), len(service_set)+len(need_pred_port_list)), order='F', dtype=np.float32)
            port_neg = np.zeros_like(port_pos)
            port_map = {}
            for index, service_tuple in enumerate(service_set):
                pS_col = (int(service_tuple[0])-1) * (len(service_to_index)+1) + int(service_to_index[str(service_tuple[1])]) + 1
                if port_matrix_first_row[0, pS_col] == 0:
                    invalid_feat.add(service_tuple)
                    continue
                temp_col = port_positive.getcol(pS_col).toarray().reshape(-1).astype(np.float32)
                temp_neg_col = port_negative.getcol(pS_col).toarray().reshape(-1).astype(np.float32)
                for i in range(len(need_pred_port_list)):
                    port_pos[i*len(service_to_index):(i+1)*len(service_to_index), index] = temp_col[start_row[i]:end_row[i]]
                    port_neg[i*len(service_to_index):(i+1)*len(service_to_index), index] = (port_matrix_first_row[0, pS_col] * port_denominator[start_row[i]:end_row[i]] - 
                                                                                            temp_neg_col[start_row[i]:end_row[i]])
                port_map[service_tuple] = index

            for index in range(len(need_pred_port_list)):
                p_col = (need_pred_port_list[index]-1) * (len(service_to_index)+1) + 1
                if port_matrix_first_row[0, p_col] == 0:
                    invalid_feat.add(need_pred_port_list[index])
                    continue
                temp_col = port_positive.getcol(p_col).toarray().reshape(-1).astype(np.float32)
                temp_neg_col = port_negative.getcol(p_col).toarray().reshape(-1).astype(np.float32)
                for i in range(len(need_pred_port_list)):
                    port_pos[i*len(service_to_index):(i+1)*len(service_to_index), index+len(service_set)] = temp_col[start_row[i]:end_row[i]]
                    port_neg[i*len(service_to_index):(i+1)*len(service_to_index), index+len(service_set)] = (port_matrix_first_row[0, p_col] * port_denominator[start_row[i]:end_row[i]] - 
                                                                                                            temp_neg_col[start_row[i]:end_row[i]])
                port_map[need_pred_port_list[index]] = index+len(service_set)

            del port_positive
            del port_negative, port_matrix, port_denominator

            if (data_dir / predictor.utils.SCANNED_TIMES_FILENAME).exists():
                old_subnet_scanned_times = scipy.sparse.load_npz(data_dir / predictor.utils.SCANNED_TIMES_FILENAME).toarray().reshape(-1).astype(np.int64)

        app_num_dict = UD(name=self.shared_dict_name)
        app_pos_first_col = app_positive_rate.getcol(0).toarray().reshape(-1).astype(np.float32)
        app_neg_first_col = app_negative_rate.getcol(0).toarray().reshape(-1).astype(np.float32)

        
        app_fc_new_index = np.zeros(len(need_pred_port_list)*len(service_to_index), dtype=np.float32)
        port_fc_new_index = np.zeros_like(app_fc_new_index, dtype=np.float32)
        port_neg_fc_new_index = np.zeros_like(app_fc_new_index, dtype=np.float32)
        app_neg_fc_new_index = np.zeros_like(app_fc_new_index, dtype=np.float32)
        for i in range(len(need_pred_port_list)):
            port_fc_new_index[i*len(service_to_index):(i+1)*len(service_to_index)] = port_positive_first_col[start_row[i]:end_row[i]]
            app_fc_new_index[i*len(service_to_index):(i+1)*len(service_to_index)] = app_pos_first_col[start_row[i]:end_row[i]]
            port_neg_fc_new_index[i*len(service_to_index):(i+1)*len(service_to_index)] = port_negative_first_col[start_row[i]:end_row[i]]
            app_neg_fc_new_index[i*len(service_to_index):(i+1)*len(service_to_index)] = app_neg_first_col[start_row[i]:end_row[i]]


        with open(self.input_dir / 'subnet' / subnet / 'app_set.pkl', 'rb') as f:
            app_set = pickle.load(f)

        app_pos = np.zeros((len(need_pred_port_list)*len(service_to_index), len(app_set)), order='F', dtype=np.float32)
        app_neg = np.zeros_like(app_pos)
        app_map = {}
        new_app_set = set()
        for index, app in enumerate(app_set):
            ori_index = app_num_dict.data.get(app, None)
            if ori_index is not None:
                temp_col = app_positive_rate.getcol(ori_index).toarray().reshape(-1).astype(np.float32)
                temp_neg_col = app_negative_rate.getcol(ori_index).toarray().astype(np.float32).reshape(-1)
                for i in range(len(need_pred_port_list)):
                    app_pos[i*len(service_to_index):(i+1)*len(service_to_index), index] = temp_col[start_row[i]:end_row[i]]
                    app_neg[i*len(service_to_index):(i+1)*len(service_to_index), index] = (app_positive_count[ori_index] * app_denominator[start_row[i]:end_row[i]] - 
                                                                                           temp_neg_col[start_row[i]:end_row[i]])
                app_map[app] = index
            else:
                new_app_set.add(app)


        with open(subnet_dir / 'infos.json', 'r') as f:
            output_dir = self.output_dir / 'subnet' / subnet
            output_dir.mkdir(exist_ok=True, parents=True)
            lines = f.readlines()
        
        output_string = ''
        ips = []
        all_feature_cols = []
        all_app_indexes = []
        all_predicted_service = []
        all_candidate_row_index = []
        all_ports = []

        for i in range(len(lines)):
            line = lines[i]
            info = dict(json.loads(line))
                        
            ips.append(info['ip'])
            ports = set(info['ports'])
            feature_cols = []
            candidate_row_index = set()
            for item in info['PortService']:
                if tuple(item) not in invalid_feat:
                    feature_cols.append(port_map[tuple(item)])
                ports.discard(item[0])
            ports = list(ports)
            all_ports.append(ports)
            for port in ports:
                if port not in invalid_feat:
                    feature_cols.append(port_map[port])
                candidate_row_index.add(port_to_index_dict[port])

            predicted_service = Dict.empty(key_type=types.int64, value_type=types.int64[:])
            for p, services in info['predicted_services'].items():
                if p in candidate_row_index:
                    predicted_service[port_to_index_dict[p]] = np.array(list(map(service_to_index.get, services)), dtype=int)

            candidate_row_index = list(candidate_row_index)


            all_feature_cols.append(np.array(feature_cols, dtype=int))
            all_predicted_service.append(predicted_service)
            all_candidate_row_index.append(np.array(candidate_row_index, dtype=int))
            all_app_indexes.append(np.array(list(filter(lambda x: x is not None, map(app_map.get, info['other_layer'].values()))), dtype=int))

        all_feature_cols.append(np.zeros(0, dtype=int))
        temp_dict = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        temp_dict[0] = np.zeros(0, dtype=int)
        all_predicted_service.append(temp_dict)
        all_candidate_row_index.append(np.zeros(0, dtype=int))
        all_app_indexes.append(np.zeros(0, dtype=int))
        total_scanned_times = np.sum(old_subnet_scanned_times, dtype=np.float32)
        uncertainty = np.zeros(len(port_positive_first_col), dtype=np.float32)
        if total_scanned_times >= 1 and self.uncertainty_weigh > 0:
            uncertainty = np.sqrt(2 * np.log(total_scanned_times) / (old_subnet_scanned_times + 1))
            uncertainty *= np.array(self.uncertainty_weigh, dtype=np.float32)

        results = calc(all_candidate_row_index, all_feature_cols,
                        all_app_indexes, all_predicted_service, app_pos, port_pos, app_neg, port_neg,
                        port_fc_new_index, app_fc_new_index, port_neg_fc_new_index, app_neg_fc_new_index,
                        len(service_to_index), self.predict_port_num, uncertainty)

        subnet_scanned_times = np.zeros(65535*len(service_to_index)+1, dtype=int)
        for i in range(len(ips)):
            ports = all_ports[i]
            result = {}
            for j in range(len(ports)):
                result[ports[j]] = []
                for service_i in results[i][j]:
                    result[ports[j]].append(index_to_service[service_i])
                    subnet_scanned_times[(ports[j] - 1) * len(service_to_index) + service_i] += 1
            output_string += (json.dumps({'ip':ips[i], 'PortService':result}) + '\n')
        fp = open(output_dir / predictor.utils.PREDICT_RESULT_FILENAME, 'w')
        fp.write(output_string)
        fp.close()
        if data_dir.exists():
            scipy.sparse.save_npz(data_dir / predictor.utils.SCANNED_TIMES_FILENAME, 
                                  scipy.sparse.csr_matrix(old_subnet_scanned_times.astype(np.int64) + subnet_scanned_times))
        time2 = time.time()
        with open(output_dir / predictor.utils.PREDICT_USING_TIME_FILENAME, 'w') as fo:
            fo.write(str(time2-time1)+'\n')

        task_queue.put((True, subnet_scanned_times, new_app_set))
        return (True, None, None)
      
                



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

        global task_queue, app_positive_rate, service_to_index, index_to_service, app_negative_rate, app_denominator, app_positive_count

        with open(self.data_dir / feature_extractor.utils.APP_NUM_DICT_FILENAME, 'rb') as f:
            app_num_dict = pickle.load(f)

        self.shared_dict_name = 'app_num'
        shared_app_num_dict = UD(app_num_dict, name=self.shared_dict_name, create=True)

        self.app_feature_num = len(app_num_dict) + 1

        with open(self.data_dir / feature_extractor.utils.SERVICE_TO_INDEX_FILENAME, 'r') as f:
            service_to_index = json.load(f)
        with open(self.data_dir / feature_extractor.utils.INDEX_TO_SERVICE_FILENAME, 'r') as f:
            temp = json.load(f)
            index_to_service = {}
            for k, v in temp.items():
                index_to_service[int(k)] = v


        with open(self.data_dir / 'app_matrix.npz', 'rb') as f:
            app_matrix = scipy.sparse.load_npz(f)
        with open(self.data_dir / 'positive_rate.npz', 'rb') as f:
            app_positive_rate = scipy.sparse.load_npz(f).tocsc()
        with open(self.data_dir / 'negative_rate.npz', 'rb') as f:
            app_negative_rate = scipy.sparse.load_npz(f).tocsc()
        

        m_row = list(range(app_positive_rate.shape[0]))
        m_col = [0] * len(m_row)
        m_data = [1] * len(m_row)
        m_row.extend(list(range(1, app_positive_rate.shape[0])))
        m_col.extend(list(range(1, app_positive_rate.shape[0])))
        m_data.extend([-1] * (app_positive_rate.shape[0] - 1))

        app_negative_rate[:, 0] = scipy.sparse.coo_matrix(
            (m_data, (m_row, m_col)), shape=(app_positive_rate.shape[0], app_positive_rate.shape[0])
            ).tocsc().dot(app_positive_rate.getcol(0))
        app_denominator = 1 / scipy.sparse.coo_matrix(
            (m_data, (m_row, m_col)), shape=(app_positive_rate.shape[0], app_positive_rate.shape[0])
            ).tocsc().dot(app_matrix.getcol(0)).toarray().reshape(-1).astype(np.float32)
        app_positive_count = app_matrix.getrow(0).toarray().reshape(-1)

        self.all_scanned_times = np.zeros(65535*len(service_to_index)+1, dtype=int)
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
            adder_process = multiprocessing.Process(target=adder, args=(result_queue,65535*len(service_to_index)+1))
            adder_process.start()

        print('Predictor: Predicting with different subnets')
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)
        pool = multiprocessing.Pool(self.processes-adders_count)
        for _ in pool.imap_unordered(self.predict_subnet, args, chunksize=self.chunk_size):
            pbar.update()
        pool.close()
        pool.join()

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
