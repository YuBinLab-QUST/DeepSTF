# DeepSTF
DeepSTF: Predicting transcription factor binding sites by interpretable deep neural networks combining sequence and shape


# Dependencies

DeepSTF works under Python 3.7

The required dependencies for DeepSTF are as follows：

python ==3.7.15

pytorch==1.10.2

numpy==1.21.5

pandas==1.3.5

scikit-learn==1.0.2

# Input

DeeepSTF takes two files as input: the Sequence file and the Shape file. The Sequence file is composed of two CSV files: one for training validation and one for testing. The datasets are available at http://cnn.csail.mit.edu/motif discovery/.The Shape file is computed from the corresponding DNA sequences in the Sequence file by the DNAshapeR tool, which can be downloaded from http://www.bioconductor.org/. The Shape file consists of ten CSV files of helix twist (HelT), minor groove width (MGW), propeller twist (ProT), rolling (Roll), and minor groove electrostatic potential (EP) for training validation data and testing data.

# Output

The "wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk" dataset is shown in the code as a sample. The train.py file is run to train the data and obtain the results. The model and its parameters are stored in deepstf.pth.
