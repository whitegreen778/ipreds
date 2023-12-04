# IPREDS (Port)

This is the branch for **port** prediction. For service prediction, please switch to the **other** branch.

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
  * example:`{"total_active_ports": 28715, "positive_true_count": 27161, "total_predict_ports": 91172, "coverage": 0.9458819432352429}`

* `input_data`: Prediction results of all cycle with host features.
* `model`: Scanning results of seed addresses, Model matrices.
* `prediction_result`: Prediction results of the latest cycle.





