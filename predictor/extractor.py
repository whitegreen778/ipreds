from feature_extractor.extractor import ExtractorOfFile
import predictor.utils as utils
import json
from tqdm import tqdm
from pathlib import Path
import pickle
import multiprocessing

class CensysExtractorForTest(ExtractorOfFile):
    def subnet_extract(self, subnet_dir: Path):
        service_set = set()
        app_feature_set = set()
        need_pred_port = set()
        max_port_count = 0
        max_app_count = 0
        line_count = 0
        with open(subnet_dir / 'infos.json', 'r') as f:
            while line := f.readline():
                temp = json.loads(line.strip())
                line_count += 1
                ports = set(temp['ports'])
                for pS in temp['PortService']:
                    service_set.add(tuple(pS))
                    ports.discard(pS[0])
                need_pred_port.update(ports)
                max_port_count = max(max_port_count, len(temp['PortService']))
                app_count = 0
                for item in self.using_app_feature:
                    if temp['other_layer'].get(item, None) is not None:
                        app_feature_set.add(temp['other_layer'][item])
                        app_count += 1
                max_app_count = max(max_app_count, app_count)
        with open(subnet_dir / utils.SUBNET_SERVICE_SET_FILENAME, 'wb') as f:
            pickle.dump(service_set, f)
        with open(subnet_dir / utils.SUBNET_APP_SET_FILENAME, 'wb') as f:
            pickle.dump(app_feature_set, f)
        with open(subnet_dir / utils.SUBNET_NEED_PRED_PORT_FILENAME, 'wb') as f:
            pickle.dump(list(need_pred_port), f)
        with open(subnet_dir / utils.SUBNET_MAX_APP_COUNT_FILENAME, 'w') as f:
            f.write(str(max_app_count))
        with open(subnet_dir / utils.SUBNET_MAX_PORT_COUNT_FILENAME, 'w') as f:
            f.write(str(max_port_count))
        with open(subnet_dir / utils.LINE_COUNT_FILENAME, 'w') as f:
            f.write(str(line_count))
        return 

    def known_subent_extract(self, using_app_feature:set):
        self.using_app_feature = using_app_feature

        args = []
        for subnet_dir in (self.dir / 'subnet').iterdir():
            if subnet_dir.is_dir():
                args.append(subnet_dir)
            
        pbar = tqdm(total=len(args))
        pool = multiprocessing.Pool(processes=self.processes)

        for _ in pool.imap_unordered(self.subnet_extract, args, chunksize=10):
            pbar.update()
            

