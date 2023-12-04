from feature_extractor.extractor import ExtractorOfFile
import json
import ipaddress
from pathlib import Path
import pickle
from tqdm import tqdm

class CensysExtractorForEvaluation(ExtractorOfFile):
    def extract(self):
        lines = []
        for file in tqdm(list(Path.iterdir(self.dir / 'input'))):
            with open(file, 'r') as f:
                while line := f.readline():
                    lines.append(line)
                    if len(lines) == self.line_vol:
                        results = self.multiprocesses_extract(lines)
                        subnet_dict = {}
                        for result in results:
                            self.combine_dict(subnet_dict, result)
                        self.multi_process_save_json(subnet_dict)
                        del lines, subnet_dict
                        lines = []
        if len(lines) > 0:
            results = self.multiprocesses_extract(lines)
            subnet_dict = {}
            for result in results:
                self.combine_dict(subnet_dict, result)
            self.multi_process_save_json(subnet_dict)
    
