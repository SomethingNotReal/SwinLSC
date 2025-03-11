## Description

SwinLSC (Swin Transformer for Lithology Section Classification) is a deep learning model for identifying micro pore combination types in thin sections of rock castings. This model is based on the Swin Transformer proposed by Microsoft Research (paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows)， Through improvement, the performance of the model in extracting microscopic morphological features has been enhanced. On the basis of the original hierarchical visual Transformer framework, this project introduces a global attention mechanism to enhance the model's perception ability of cross regional features, integrates channel attention modules to achieve adaptive weight adjustment of feature channels, and adds an end convolution processing module to the backend of the Transformer. The basic architecture of the model is derived from Microsoft Research's open-source project（ https://github.com/microsoft/Swin-Transformer ）Implement with relevant communities（ https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer ）On this basis, we have made targeted improvements and optimizations to make it more suitable for the identification task of micro pore combination types in rock thin sections.

Due to the fact that the dataset used in this research is an internal dataset of the oil group and involves commercial secrets, it is not open source. In this project, if necessary, the Flower dataset can be used（ https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz ）Perform training prediction. The pre training weights provided in this project are from Imagenet-1K-miniTraining preservation（ https://modelscope.cn/datasets/tany0699/mini_imagenet100 ）.

## Requirements

- torch
- numpy
- PIL
- scipy
- pandas
- scikit-learn
- matplotlib

## Usage



Train. py and SwinLSC_model. py are the core files of the project. Please follow the following steps to run the project：

1. Ensure that all project files have been downloaded completely

2. Ensure that the model is imported into the train.py file. You can import models of different depths, such as: small，tiny，base，large. Please refer to SwinLSC_modl.py for details. The above import method is also applicable in ablation experiments, using the same train.py to import different models in the ablation experiment.

3. ```python
   from SwinLSC_model import swin_tiny_patch4_window7_224 as create_model
   ```

4. Ensure that the dataset is classified correctly and that images of the same category are stored in a folder named after the category.

6. A reasonable set of input parameters are as follows:

7. ```python
       # Parse command line arguments
       parser = argparse.ArgumentParser()
       parser.add_argument('--num_classes', type=int, default=5)#Number of categories that need to be classified
       parser.add_argument('--epochs', type=int, default=100)#Training epochs
       parser.add_argument('--batch-size', type=int, default=8)
       parser.add_argument('--lr', type=float, default=0.0001)#Learning rate
   
       # Root directory of the dataset
       # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
       parser.add_argument('--data-path', type=str,
                           default="Your image directory location")#Enter the directory path, such as:(./flower_photos) or (D:\\Users\\flower_photos)
   
       # Path to pre-trained weights, if not loading, set to empty string
       parser.add_argument('--weights', type=str, default='/imagenet-1k_weights/SwinLSC_imagenet.pth/imagenet-1k_weights/SwinLSC_imagenet.pth',
                           help='initial weights path')#Input pre training weights, such as./imagenet-1k_weights/SwinLSC_imagenet.pth
       # Whether to freeze weights
       parser.add_argument('--freeze-layers', type=bool, default=False)
       parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
   ```

8. After ensuring the above configuration is correct, run train.by in the IDE or type "python train.py" in the terminal to start training.



## Related publications

If you use SwinLSC in your work, please consider citing one or more of these publications:

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021b. Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022. https://doi.org/10.48550/arXiv.2103.14030.

Vaswani, A., 2017. Attention is all you need. Advances in Neural Information Processing Systems https://doi.org/10.48550/arXiv.1706.03762.

Wang, S., Li, B.Z., Khabsa, M., Fang, H., Ma, H., 2020. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 https://doi.org/10.48550/arXiv.2006.04768.

## License

SwinLSC is licensed under the [Apache License 2.0](https://github.com/zsylvester/meanderpy/blob/master/LICENSE.txt).
