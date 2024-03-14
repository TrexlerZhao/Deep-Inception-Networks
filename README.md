# Deep-Inception-Networks
This code is a version of implement of the essay named Deep Inception Networks: A General End-to-End Framework for Multi-asset Quantitative Strategies.
The code compared to the original structure of the essay aboved has several changes. In the FlexCIM feature extractor module, the convolution kernel is all the same no matter how many layers in the loops.
Also the loss function consists of several versions such as position penalization and return indicator besides the sharpe ratio.


Moreover, the input is the Chinese stock data, and after the preprocessing, the raw data is converted into daily return scaled by the exponential moving average standard deviation.


The followng part is the introduction of different python files. Most of the files are the same as the file name suggests. return function.py is used to make preparations for the loss function part, and the train_test_function.py is leveraged to pack up all the training and testing procedure into several funtions. Therefore it simplifies the code complexity but hard to debug. FlexCIM result2 is the final execution file, including the data preprocessing, train-test-split part, and traing-testing part.


Please to be noted that, all the feature extraction part only includes FlexCIM network structure. Also, all the backtest is applied in the manner of moving test, that means after every loop(different time period put into the same model, using the moving window method), we extract the last time step of the output which denotes the position predicted of the last day. We later append all of them so that we get the result.
I will update a Jupyter Notebook version in a few days.
