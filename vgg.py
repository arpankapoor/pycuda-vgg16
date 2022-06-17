#!/usr/bin/env python3
import datetime
import sys
import h5py
import numpy as np

from PIL import Image

# from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#file-imagenet1000_clsidx_to_labels-txt
from labels import IMAGENET_LABELS


CUDA_SRC = """
__global__ void conv2d(const float *inp, float *out, const float *w,
                       const float *b, int inp_channels, int ksize) {
  int channel_num = threadIdx.x;
  int out_channels = blockDim.x;

  // pixel (x, y)
  int x = blockIdx.x;
  int y = blockIdx.y;

  int outidx = out_channels * gridDim.y * x +
               out_channels * y +
               channel_num;

  for (int i = 0; i < ksize; i++) {
    for (int j = 0; j < ksize; j++) {
      for (int k = 0; k < inp_channels; k++) {

        // w is 4D with dimensions: (ksize, ksize, input_channels, output_channels)
        int widx = (ksize * inp_channels * out_channels * i) +
                   (inp_channels * out_channels * j) +
                   (out_channels * k) +
                   channel_num;

        // inp is 3D with dimensions: (blockDim.x + padding, blockDim.y + padding, input_channels)
        int inpidx = ((gridDim.y + ksize - 1) * inp_channels * (i + x)) +
                     (inp_channels * (j + y)) +
                     k;

        out[outidx] += inp[inpidx] * w[widx];
      }
    }
  }

  // add bias
  out[outidx] += b[channel_num];

  // relu
  if (out[outidx] < 0) {
    out[outidx] = 0.0;
  }
}
"""


# from https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/applications/imagenet_utils.py#L168-L238
def preprocess_img(img):
    x = img.astype(np.float32)
    # 'RGB'->'BGR' (because of opencv?)
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


# from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
VGG_WTS = h5py.File("vgg16_weights_tf_dim_ordering_tf_kernels.h5", "r")


# layer names from https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/vgg16.py#L43-L227
# can also be obtained by executing `model.summary()` on the `Model` object
VGG_LAYERS = ['block1_conv1', 'block1_conv2', 'block1_pool',
              'block2_conv1', 'block2_conv2', 'block2_pool',
              'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
              'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
              'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool',
              'flatten',
              'fc1', 'fc2',
              'predictions']


# return top predictions with probabilities from output of last layer
def get_top_predictions(preds, top=5):
    return {IMAGENET_LABELS[x]: preds[x] for x in (-preds).argsort()[:top]}


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def applyConv2d(w, b, inp, cuda=False):
    """assuming odd-sized square kernel"""
    ksize = w.shape[0]
    inp_channels = inp.shape[-1]
    out_channels = w.shape[-1]

    # pad the input, not output: https://stackoverflow.com/a/69544897
    padded_inp = np.pad(inp, ((ksize//2, ksize//2),
                              (ksize//2, ksize//2),
                              (0, 0)))

    # output is same shape as input with more channels
    out = np.zeros(inp.shape[:2] + (out_channels,), dtype=np.float32)

    if cuda:
        # convert numpy arrays to pycuda.gpuarray
        in_gpu, out_gpu = pycuda.gpuarray.to_gpu(padded_inp), pycuda.gpuarray.to_gpu(out)
        w_gpu, b_gpu = pycuda.gpuarray.to_gpu(w), pycuda.gpuarray.to_gpu(b)

        # call cuda implementation with the GPU arrays
        cudaConv2d(in_gpu, out_gpu, w_gpu, b_gpu, np.int32(inp_channels), np.int32(ksize),
                   block=(out_channels, 1, 1), grid=inp.shape[:2])

        # copy back the output from GPU memory
        return out_gpu.get()
    else:
        # convolve the surrounding of each pixel (3x3x3) with the kernel
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                for c in range(out_channels):
                    out[x][y][c] = np.tensordot(w[..., c],
                                                padded_inp[x:x+ksize, y:y+ksize],
                                                ksize)

        # add bias to each output channel
        for c in range(out_channels):
            out[..., c] += b[c]

        # apply relu activation
        return relu(out)


def applyMaxPool2d(inp):
    """2x2 kernel with (2,2) stride"""
    out = np.zeros((inp.shape[0]//2, inp.shape[1]//2, inp.shape[2]), dtype=np.float32)
    for x in range(out.shape[0]):
        for y in range(out.shape[1]):
            for c in range(inp.shape[2]):
                out[x][y][c] = np.max(inp[2*x:2*(x+1), 2*y:2*(y+1), c])
    return out


def applyVgg16(inp, cuda=False):
    curr = inp
    outputs = []
    for layer in VGG_LAYERS:
        if "pool" in layer:
            out = applyMaxPool2d(curr)
        elif layer == "flatten":
            out = curr.flatten()
        # weight layers
        else:
            w, b = (np.array(x) for x in VGG_WTS[layer].values())
            if layer.startswith("fc") or layer == "predictions":
                # fully connected layers are a simple matrix multiplication
                out = np.matmul(w.T, curr.reshape(-1, 1)).flatten() + b
                out = relu(out) if layer.startswith("fc") else softmax(out)
            else:
                out = applyConv2d(w, b, curr, cuda)
        outputs.append(out)
        print(f"processed {layer}: inshape: {curr.shape}, outshape: {out.shape}")
        curr = out

    # return output of all hidden layers along with output layer
    # helps in inspecting output of hidden layers
    return outputs



if __name__ == "__main__":
    # arg1 = tf/numpy/cuda
    # arg2 = image path

    # read sample image using PIL and resize to the size required by VGG16
    img = np.asarray(Image.open(sys.argv[2]).resize((224, 224)))
    
    # preprocess input image as done by tensorflow
    img = preprocess_img(img)

    if sys.argv[1] == "tf":
        import tensorflow as tf

        model = tf.keras.applications.vgg16.VGG16()
        start = datetime.datetime.now()
        outputs = [model.predict(img[np.newaxis, ...])[0]]
        end = datetime.datetime.now()
    elif sys.argv[1] == "numpy":
        start = datetime.datetime.now()
        outputs = applyVgg16(img)
        end = datetime.datetime.now()
    else:
        import pycuda.gpuarray
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        mod = SourceModule(CUDA_SRC)
        cudaConv2d = mod.get_function("conv2d")

        start = datetime.datetime.now()
        outputs = applyVgg16(img, cuda=True)
        end = datetime.datetime.now()

    print("predictions: ", get_top_predictions(outputs[-1]))
    print(f"computed in {end-start}")
