# Mind-Gut-Connection

CNN and RCNN implementations to perform classification of stethoscope sounds based on different mental states.

6 Classes are considered:

1. Anxiety
2. Baseline
3. Concentration
4. Digestion
5. Disgust
6. Frustration

## Notebooks

Jupyter notebooks are used to explore and unit test data before deploying to a model

##### EDA - Exploratory data analysis

Plots time series of 1m data points (fs=48,000). Signals are down sampled to 8k Hz.
Calculates and displays frequencies using the Fast Fourier Transform.
Plots the Short Time Fourier Transform for an interval of 3 seconds and truncates to frequencies under 1k hertz.
The corresponding Mel Frequency Cepstral Coefficients are calculated as well uses 13 mel filters scaled from 0 to 4k hertz.

##### Downsample

Down samples all audio to 8k hertz and saves wavfiles in new directory with respective class.

##### Demo high pass filter and noise threshold

Applies a high pass filter of 10 Hz to the 8k hertz signal and applies a noise threshold using a moving average of 1 second.
Portions of the signal to keep and discard are shown in green and red.

##### Calculates STFT and MFCC features

Builds STFT and MFCC features separated as 1 second intervals. A high pass filter and noise threshold are applied the the signal.
Features are saved as npz files.

##### Input Scaling

Looks at the min and max values for each sample in either stft or mfcc directory.
Considers log10(features)+6 as a lambda function transform as input to the neural network for a normal distribution.

##### Evaluate Subjects

Predicts accuracies for each subject in either stft or mfcc directories.
Also has some results from cross validation on both ordered and unordered subjects.

## Running the models

Two models are implemented. Convolutional Neural Net (CNN). Recurrent Convolutional Neural Net. (RCNN)

Arguments to model are found in main.py.
Total duration should be integers of 1 second intervals.
Delta time will slice the total duration into time intervals for the RCNN. (ignored for CNN)

`python main.py`

Trains the model using 25% percent of the training set per epoch. Shuffles each epoch. (9/1 train test split)

**Note** Check the output size of each convolution block if you are making delta time smaller.
Comment out blocks if output shape into the LSTM to too small.
