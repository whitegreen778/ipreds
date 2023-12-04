import feature_extractor.extractor
import feature_extractor.censys
from pathlib import Path
from predictor.predictor import Predictor
from evaluator.evaluator import Evaluator
from updater.updater import Updater
import gen_model_data.extractor as ex
import shutil
from UltraDict import UltraDict as UD
import time
from tqdm import tqdm


# Please change the following arguments to suit your own environment
env_path = Path('./')
groundtruth_path = Path('./ground_truth')
processes = 3
app_feature = ['asn', 'organization']
pre = True
prediction_cycle = 7
port_num_per_cycle = 4
uncertainty_weight = 0
####################################################################



assert processes >= 2
input_path = env_path / 'input_data'
model_path = env_path / './model'
predict_path = env_path / './predict_result'
positive_truth_path = env_path / './eval_result'
adders = 1
chunk_size = processes - adders
a = UD(name='app_num')
a.unlink()

if True and pre:
    print('Deleting Model')
    if model_path.exists():
        shutil.rmtree(model_path)
    (model_path / 'subnet').mkdir(exist_ok=True, parents=True)

    extractor = ex.ModelSelector(groundtruth_path / 'subnet', model_path / 'subnet',
                                0.1, app_feature, processes)
    extractor.select_data()

if True and pre:
    print('Deleting old model')
    delete_list = []
    delete_list.extend(list((model_path / 'subnet').rglob('*.npy')))
    delete_list.extend(list((model_path / 'subnet').rglob('*.npz')))
    delete_list.extend(list((model_path / 'subnet').rglob('*.pkl')))
    delete_list.extend(list((model_path / 'subnet').rglob('*.txt')))
    for item in tqdm(delete_list, ncols=80, position=1, ascii=True):
        item.unlink()

extractor = feature_extractor.censys.CensysFileExtractor(dir=model_path, processes=processes)
saver = feature_extractor.extractor.FeatureSaver(model_path, extractor, subnet_threads=processes, chunk_size=chunk_size, adders=adders)

if True and pre:
    saver.extract(using_app_feature=app_feature)
    saver.calc()
    saver.calc_app()

updater = Updater(combined_dir=input_path / 'subnet', positive_true_dir=positive_truth_path, model_dir=model_path / 'subnet',
                    app_matrix_dir=Path(model_path / 'app_matrix.npz'),
                    app_num_dir=Path(model_path / 'app_num_dict.pkl'),
                    predict_result_dir=predict_path  / 'subnet', chunk_size=chunk_size,
                    using_app_feature=app_feature, processes=processes, groundtruth_dir=groundtruth_path / 'subnet', adders=adders)
predictor = Predictor(input_path, model_path, predict_path, extractor_processes=processes, adders=adders,
                    processes=processes, predict_port_num=port_num_per_cycle, uncertainty_weigh=uncertainty_weight, chunk_size=chunk_size)
evaluator = Evaluator(groundtruth_path, predict_path / 'subnet', positive_truth_output=positive_truth_path,
                processes=processes, extractor_processes=processes, chunk_size=chunk_size)

if True and pre:
    print('Deleting predict result')
    if predict_path.exists():
        shutil.rmtree(predict_path)
    (predict_path /'subnet').mkdir(exist_ok=True, parents=True)

    print('Deleting eval output')
    if positive_truth_path.exists():
        shutil.rmtree(positive_truth_path)
    positive_truth_path.mkdir(exist_ok=True, parents=True)

if True and pre:
    print('Deleting input')
    if input_path.exists():
        shutil.rmtree(input_path)
    (input_path / 'subnet').mkdir(exist_ok=True, parents=True)
    updater.update(init=True)


for _ in range(prediction_cycle):
    time1 = time.time()
    predictor.predict(using_app_feature=app_feature)
    time2 = time.time()
    with open(positive_truth_path / 'predict_time.txt', 'a') as f:
        f.write(str(time2-time1)+'\n')

    evaluator.evaluate()

    time3 = time.time()
    updater.update(init=False)
    time4 = time.time()
    with open(positive_truth_path / 'update_time.txt', 'a') as f:
        f.write(str(time4-time3)+'\n')
