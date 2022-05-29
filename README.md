A VGG16 inference module comparison between
- tensorflow
- numpy
- pycuda

# execute

- required dependencies were installed with miniforge using `conda install pillow pycuda tensorflow-gpu`
- VGG16 weights pretrained on ImageNet were downloaded from [here](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5)
  and placed in the same directory as the code in this repository

```console
$ ./vgg.py tf /path/to/image_to_classify      # run inference using tensorflow
$ ./vgg.py numpy /path/to/image_to_classify   # run inference using numpy implementation
$ ./vgg.py pycuda /path/to/image_to_classify  # run inference using pycuda implementation
```
