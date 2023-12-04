from pathlib import Path
import multiprocessing
from tqdm import tqdm
import json
from evaluator.extractor import CensysExtractorForEvaluation
import evaluator.utils as utils
import predictor.utils

class Evaluator:
    def __init__(self, groundtruth_dir:Path, predict_dir:Path,
                processes=4, extractor_processes=4,
                positive_truth_output=None, chunk_size=25):
        self.groudtruth_dir = groundtruth_dir
        self.predict_dir = predict_dir
        self.processes = processes
        self.positive_truth_output = positive_truth_output
        self.chunk_size = chunk_size
        self.file_extractor = CensysExtractorForEvaluation(groundtruth_dir, extractor_processes)
        
    
    def arg_wrapper_evaluate_subnet(self, args):
        return self.evaluate_subnet(*args)

    def evaluate_subnet(self, predict_dir:Path, groundtruth_dir:Path):
        total_active_ports = 0
        positive_true_count = 0
        total_predict_ports = 0
        if (predict_dir / 'result.json').exists():
            groundtruth = {}
            with open(groundtruth_dir / 'infos.json', 'r') as f:
                while line := f.readline():
                    temp = json.loads(line.strip())
                    if groundtruth.get(temp['ip'], None) is not None:
                        groundtruth[temp['ip']].update(temp['ports'])
                    else:
                        groundtruth[temp['ip']] = set(temp['ports'])
                    del temp


            output_file = None
            if self.positive_truth_output is not None:
                output_dir = self.positive_truth_output / predict_dir.name
                Path.mkdir(output_dir, exist_ok=True, parents=True)
                output_file = open(output_dir / utils.POSITIVE_TRUE_FILENAME, 'w')
            with open(predict_dir / 'result.json', 'r') as f:
                predict_result = {}
                while line := f.readline():
                    temp = json.loads(line)
                    if predict_result.get(temp['ip'], None) is not None:
                        predict_result[temp['ip']].update(temp['ports'])
                    else:
                        predict_result[temp['ip']] = set(temp['ports'])
                    del temp

            for key, value in predict_result.items():
                total_active_ports += len(groundtruth[key])
                total_predict_ports += len(value)
                positive_true_count += len(value & groundtruth[key])
                output_file.writelines(json.dumps({'ip':key, 'ports':list(value & groundtruth[key])}) + '\n')

            if total_active_ports == 0:
                coverage = 0
            else:
                coverage = positive_true_count / total_active_ports
            if total_predict_ports == 0:
                hitrate = 0
            else:
                hitrate = positive_true_count / total_predict_ports

            if output_file is not None:
                output_file.close()
                with open(predict_dir / predictor.utils.PREDICT_USING_TIME_FILENAME, 'r') as fi:
                    with open(output_dir / predictor.utils.PREDICT_USING_TIME_FILENAME, 'a') as fo:
                        fo.write(fi.read().strip() + '\n')
                with open(output_dir / utils.METRIC_FILENAME, 'a') as f:
                    f.write(json.dumps({'total_active_ports':total_active_ports, 'positive_true_count':positive_true_count,
                                'total_predict_ports':total_predict_ports, 'coverage':coverage}) + '\n')
            
        return total_active_ports, total_predict_ports, positive_true_count





    def evaluate(self):

        args = []
        pool = multiprocessing.Pool(processes=self.processes)
        for subnet_dir in self.predict_dir.iterdir():
            if subnet_dir.is_dir():
                args.append((subnet_dir, self.groudtruth_dir / 'subnet' / subnet_dir.name))

        total_active_ports = 0
        total_predict_ports = 0
        positive_true_count = 0
        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)

        for result in pool.imap_unordered(self.arg_wrapper_evaluate_subnet, args, chunksize=50):
            total_active_ports += result[0]
            total_predict_ports += result[1]
            positive_true_count += result[2]
            pbar.update(1)

        coverage = positive_true_count / total_active_ports
        hitrate = positive_true_count / total_predict_ports

        if self.positive_truth_output is not None:
            with open(self.positive_truth_output / utils.METRIC_FILENAME, 'a') as f:
                f.write(json.dumps({'total_active_ports':total_active_ports, 'positive_true_count':positive_true_count,
                            'total_predict_ports':total_predict_ports, 'coverage':coverage}) + '\n')
        
        print('total_active_ports', total_active_ports)
        print('positive_true_count', positive_true_count)
        print('total_predict_ports', total_predict_ports)
        print('coverage', coverage)

        

        