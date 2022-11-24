# Ariel Data Challenge

Python 3.9.6 and CUDA 11.3.1.

## Contact

Ondřej Podsztavek (podszond@fit.cvut.cz)

## Report

The Python script named “ariel.py” trains models, and the Jupyter notebook named “test.ipynb” generates the required outputs.

*How did you use the training data?*

I used all the 21988 annotated spectra and their auxiliary data as training data.

I optimised hyperparameters on a separate validation set (i.e. 20 % of the annotated data) using early stopping on the light score.
However, after optimising the hyperparameters, I used all annotated data to train the final models with a fixed number of epochs.

I computed weighted means and weighted variances of traces, i.e. I fitted a normal distribution to traces.
The weighted means and weighted variances are used as targets in training.

*Did you perform any data preprocessing steps?*

I scale the spectra so that each spectrum (not each feature) has zero mean and unit variance.
This scaling standardises the intensities of each spectrum, so only the shape of each spectrum should matter.

I standardise the auxiliary data so that each feature has zero mean and unit variance.

*What kind of model did you go for?*

I used a deep ensemble (Lakshminarayanan et al., 2017) of 20 models.
Each model is a convolutional neural network (CNN).
Concretely, it is a modification of the VGG Net-A CNN (Simonyan and Zisserman, 2015).
Its convolutional part consists of 6 convolutional layers and 4 max pooling layers.
The convolutional part is followed by a fully connected neural network with 7 hidden layers with 1024 neurons.
The activation function of all convolutional and fully connected layers (except the last layer) is the rectified linear unit (ReLU).
The last layer produces 12 floating point numbers:
6 are interpreted as means of normal distributions, while the rest are interpreted as variances of the normal distributions (passed through the softplus function, and a minimal variance of 10-6 is added).

I train each model with the Kullback–Leibler divergence (between two normal distributions) as the loss function using Adam optimiser with a learning rate of 10-3 for 2048 epoch and batch size of 256.

*What is the input/output of the model?*

The input of the convolutional part is a spectrum.
The output of the convolution part is concatenated with the auxiliary data.
Such a vector is the input of the fully connected part.
The output of the fully connected part (i.e. the output of the model) is interpreted as 6 means and 6 variances of normal distributions for each target (see above).

*Did you perform any sampling steps? If so, please describe the sampler.*

I sample each of my 20 models (i.e. normal distributions that they produced) 250 times.
So, I get 5000 samples that are the output for the regular track.
Then, I compute the quartiles of those 5000 samples, and that is the output for the light track.

## Appendix

I tried much more (sample noisy data, predict full covariance matrix, pre-train with unlabelled data, train with different loss functions, add dropout and batch normalisation, optimise hyperparameters with out-of-distribution validation set etc.) but the method described above worked best on the 2nd test set.

## References

Lakshminarayanan, B., Pritzel, A., Blundell, C., 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In: Guyon, I., et al. (Eds.), Advances in Neural Information Processing Systems 30.

Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.
