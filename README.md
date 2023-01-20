# Image classification

The training procedure can be run with the following command :

```
python train.py --model_name resnet50 --rootdir "dataset" --annotations_file "dataset/train_data/annotations.csv" --results_dir "results_expx"
```

In the case of my datataset annotations csv contains two columns: 'sample' -> the relative path to the images and 'label'. By passing the rootdir option,
the relative path from the csv will be appended to the root directory. To use the same code with dataset in another format, the simplest is to replace the instantiation of the LabeledDataset class from train.py with your own dataset.

The network checkpoints, the tensorboard logs and a JSON file with the hyperparameter configuration named run_metadata.json will be saved in the **results_dir**.

Other important options that can be used are:

| Option        | Description           | Default value  |
| ------------- |:-------------:| -----:|
| model_name     | Arhitecture supported : resnet50, resnet152, densenet161, vgg19 | N/A |
| batch_size    |  Batch size     |   16 |
| epochs | Num. of epochs      |   40 |
| lr | Learning rate      |  0.001 |
| momentum | Momentum     |  0.9 |
| feature_extract | If set, all the networks parameter are freezed other than the classification layer|  false |
| freq_ckpt | Frequency of checkpointing |  5 |
| smooth_rate | Label smoothing rate |  0.0 |



The default optimizer used is SGD.

If you want to train an ensemble network ( by default it will be resnet152 + densenet161 + vgg19 ), use the following command flags:

```
  --use_ensemble --freeze_ensemble_models --use_checkpoints_ensemble \
  --resnet_ckpt $RESNET_CKPT --densenet_ckpt $DENSENET_CKPT --vgg_ckpt $VGG_CKPT
```

Credits to the following repositories :   

https://github.com/sthalles/SimCLR  

https://github.com/yikun2019/PENCIL

