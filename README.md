# File Description 
/src/data/Data_Augmentation.R

 A tool of data augmentation provided by Mao et al. (2017). 

 The details of the tool can be seen in https://github.com/wgmao/EPIANN.

 We used this tool to amplify the positive samples in the training set to 20 times to achieve class balance.

/src/sequence_processing.py

 This method is provided by Zeng et al. (2019). which can pre-processing DNA sequences. 

 The details of the tool can be seen in https://github.com/hzy95/EPIVAN/.
 
/src/embedding_matrix.npy

 This tool is provided by Ng (2017).

 The details of the tool can be seen in https://github.com/pnpnpn/dna2vec.

 We use this tool convert the pre-trained DNA vector.

 /src/train.py

Perform model training.

 /src/test.py

Evaluate the performance of model.

 /src/model

The model of EPI-MIND
 
References:

Mao, W. et al. (2017) Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks. bioRxiv, 219667.

Hong Z. et al. (2019) Identifying enhancer-promoter interactions with neural network based on pre-trained DNA vectors and attention mechanism. Bioinformatics.

Ng, P. (2017) dna2vec: Consistent vector representations of variable-length k-mers. arXiv:1701.06279.
