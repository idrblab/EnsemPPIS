### The ensemble framework of EnsemPPIS for predicting protein-protein interaction sites (PPIS):

![image](/figure/EnsemPPIS.png)

### Please follow these steps to train the EnsemPPIS for the prediction of PPIS:

1. Create a Python virtual environment (version=3.7.11) according to the `'requirement.txt'` file;

2. Put your sequence data into the `'data_cache'` folder, and run `'data_preprocess.py'` in this folder to generate three pkl files, namely, encode_data.pkl, label.pkl and dset_list.pkl;

3. Download the pre-trained ProtBERT model (pytorch_model.bin) from `http://ensemppis.idrblab.cn/download_ProtBERT`, and put it into the `'feature_generator'` folder;

4. Run `'ProtBERT_feature_generator.py'` in the `'feature_generator'` folder to generate ProtBERT enbeddings for sequences;

5. Run `'main-TransformerPPIS.py'` to train the TransformerPPIS model;

6. Run `'main-GatCNNPPIS.py'` to train the GatCNNPPIS model;

### Predict PPIS using the trained EnsemPPIS:

7. Run `'predict_EnsemPPIS.py'` to predicte PPIS using the trained TransformerPPIS and GatCNNPPIS.

### Datasets:
    
All the benchmark datasets used in this study was provided in the  `'datasets'` folder.


### Training commond:
The two base models of EnsemPPIS (TransformerPPIS and GatCNNPPIS) were trained separately.

The commond for training TransformerPPIS on CPU:

    nohup python main-TransformerPPIS.py > out-TransformerPPIS.txt 2>&1 &

The commond for training TransformerPPIS using single GPU or distributed training using multiple GPUs:
single GPU: 

    CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 6666 main-TransformerPPIS.py > out-TransformerPPIS.txt 2>&1 &

multiple GPUs:

    CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 main-TransformerPPIS.py > out-TransformerPPIS.txt 2>&1 &
