# Deep Ensemble predicts Exoplanets’ Atmospheres Composition

Standard methods for inferring planetary characteristics from exoplanets’ atmospheric spectra are slow.
We present a fast machine learning method:
a deep ensemble of convolutional neural networks (CNNs) that outputs mixtures of normal distributions for planetary characteristics.
The architecture of our CNN was inspired by VGG networks.
We train each CNN with Kullback–Leibler divergence as its loss function on simulated exoplanet’ atmospheric spectra and their auxiliary data from Ariel Data Challenge.

The Python script named `ariel.py` trains models, and the Jupyter notebook named `test.ipynb` generates the required outputs.

See `slides.pdf` for some information.

## Installation

Python 3.9.6, CUDA 11.3.1, and PyTorch 1.12.1.

    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    $ pip install -r requirements.txt

## Data

Data are available on Zenodo, at https://doi.org/10.5281/zenodo.6770103.
Concretely, it is the ZIP file `Level2Data.zip`.

## Contact

Ondřej Podsztavek (podszond@fit.cvut.cz)

## References

Lakshminarayanan, B., Pritzel, A., Blundell, C., 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In: Guyon, I., et al. (Eds.), Advances in Neural Information Processing Systems 30.

Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.
