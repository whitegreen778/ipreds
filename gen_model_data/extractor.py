from feature_extractor.extractor import ExtractorOfFile
import feature_extractor.utils
import json
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import random
import math

class ModelSelector:
    def __init__(self, input_dir: Path, output_dir: Path, ratio: float,
                using_app_feature: list, processes: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ratio = ratio
        self.using_app_feature = using_app_feature
        self.processes = processes

    def subnet_select_data(self, subnet_name: str):
        input_filename = self.input_dir / subnet_name / feature_extractor.utils.INFO_JSON_FILENAME
        output_filename = self.output_dir / subnet_name / feature_extractor.utils.INFO_JSON_FILENAME

        input_file = open(input_filename, 'r')
        lines = input_file.readlines()
        output_lines = random.sample(lines, k=math.ceil(len(lines) * self.ratio))
        (self.output_dir / subnet_name).mkdir(exist_ok=True, parents=True)
        output_str = ''

        for line in output_lines:
            temp = json.loads(line.strip())
            temp_dict = {'ip':temp['ip'], 'ports':temp['ports'], 'PortService':temp['PortService'], 'other_layer':{}}
            for feature in self.using_app_feature:
                temp_dict['other_layer'][feature] = temp[feature]
            output_str += json.dumps(temp_dict) + '\n'

        with open(output_filename, 'w') as output_file:
            output_file.write(output_str)
        

    def select_data(self):
        print('Selecting model data')
        args = []
        for subnet_dir in self.input_dir.iterdir():
            if subnet_dir.is_dir():
                args.append(subnet_dir.name)

        pbar = tqdm(total=len(args), ncols=80, position=1, ascii=True)
        pool = multiprocessing.Pool(processes=self.processes)
        for _ in pool.imap_unordered(self.subnet_select_data, args, chunksize=25):
            pbar.update()

        