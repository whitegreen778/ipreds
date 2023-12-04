from feature_extractor.extractor import ExtractorOfFile
import ipaddress
import json
from pathlib import Path
import feature_extractor.utils as utils

class CensysFileExtractor(ExtractorOfFile):

    def subnet_extract(self, subnet_dir: Path):
        port_count = 0
        app_feature_set = set()
        with open(subnet_dir / 'infos.json', 'r') as f:
            for line in f.readlines():
                temp = json.loads(line.strip())
                if self.need_port_count:
                    port_count += len(temp['ports'])
                if self.need_app_num_dict:
                    for app in self.using_app_feature:
                        feat = temp['other_layer'].get(app, None)
                        if feat is not None:
                            app_feature_set.add(feat)
        if self.need_port_count:
            with open(subnet_dir / utils.SUBNET_PORT_COUNT_FILENAME, 'w') as f:
                f.write(str(port_count))
        
        return app_feature_set
