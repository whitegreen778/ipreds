# IPREDS (Service)

This is the branch for **service** prediction. For port prediction, please switch to the **other** branch.

## Environment

Tested on Manjaro 23.0 and Ubuntu 22.04.

Make sure Anaconda or Miniconda has already been installed.

The following command installs the dependency of IPREDS:

``` bash
conda env create -f conda_env.yaml
```

Then activate the environment:

```bash
conda activate IPREDS
```

## Data

Please download the groundtruth data from the [URL](https://drive.google.com/file/d/1u-HYeHV2cUtbiJcoBkF4o3XDstgHvJD7/view?usp=sharing) and unzip it to a suitable place.

Due to ethical and privacy considerations, we only provide a limited  scope of public data for testing purposes. Users are able to employ other datasets for their testing needs.

## Getting Started

Please change the arguments in `test.py` to suit your environment

```python
# The directory to store the models and the results of prediction
# Please use a clean folder.
env_path = Path('./')

# The two files contain the corresponding relationships between services and their indexes and are mutually interrelated. 
# IPREDS will only process and predict the types of services listed in these two files. # The indexing must start from 1 and increase by 1. 
# Interchanging the indexes of any two types of services will not affect the prediction results. 
# You may add or delete service types in these files, but please ensure that all service types present in the ground truth are included.
index_to_service_path = Path('./index_to_service.json')
service_to_index_path = Path('./service_to_index.json')

# The directory containing the groundtruth data.
# Be sure to contain a subfolder named "subnet", which contains the data of each subnet.
groundtruth_path = Path('./ground_truth')

# How many processes to use during the prediction. 
# Please use at least 2 process
processes = 3

# The application layer features used in prediction
# Make sure they are contained in your groundtruth data
app_feature = ['asn', 'organization']

# Set to "True" to have a fresh start
# Set to "False" to continue the last prediction
pre = True

# How many times you want to proceed the prediction process
prediction_cycle = 7

# How many ports to predict in each prediction
port_num_per_cycle = 4

# The weight of exploration module.
# Set to 0 to disable exploration.
uncertainty_weight = 0
```

Then start the prediction with:

```bash
conda activate IPREDS
python test.py
```

You can see the metrics and the time consumption of each cycle in your console.

## Results

All the data created by IPREDS are under the `env_path`:

* `eval_result`: Positive true results, Coverage, Time consumption in seconds.
  * example:`{"total_active_services": 26692, "positive_true_count": 18825, "total_predict_services": 28715, "coverage": 0.705267495878915}`

* `input_data`: Prediction results of all cycle with host features.
* `model`: Scanning results of seed addresses, Model matrices.
* `prediction_result`: Prediction results of the latest cycle.





