
# Covid detection algorithm

Here is the code for the covid detection algorithm. It consists on 3 pre-trained Resnets combined (Transfer learning)

- Notes: For intelectual property reasons, the file MIT_models.py was not added, so the structure of the original ML models cannot be seen

## Files

- [parameters.json](https://github.com/luiscuervo/machine_learning/blob/main/covid_detection/parameters.json) --> Set up the parameters for the experiment (GPU distribution, file path, learning rate...)
- [main.py](https://github.com/luiscuervo/machine_learning/blob/main/covid_detection/main.py) --> Trains the model
- [db_utils.py](https://github.com/luiscuervo/machine_learning/blob/main/covid_detection/db_utils.py) --> Contains database handling functions
- [test_models.py](https://github.com/luiscuervo/machine_learning/blob/main/covid_detection/test_models.py) --> Trains fully tested models
