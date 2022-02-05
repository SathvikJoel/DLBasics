# Pytorch
---

With PyTorch tensors we can load and unload tensors from the GPU. 

PyTorch goes very well with Numpy.

<table class="table table-sm table-hover">
    <tbody>
        <tr>
            <th>
                Package
            </th>
            <th>
                Description
            </th>
        </tr>
        <tr>
            <td>
                torch
            </td>
            <td>
                The top-level PyTorch package and tensor library.
            </td>
        </tr>
        <tr>
            <td>
                torch.nn
            </td>
            <td>
                A subpackage that contains modules and extensible classes for building neural networks.
            </td>
        </tr>
        <tr>
            <td>
                torch.autograd
            </td>
            <td>
                A subpackage that supports all the differentiable Tensor operations in PyTorch.
            </td>
        </tr>
        <tr>
            <td>
                torch.nn.functional
            </td>
            <td>
                A functional interface that contains typical operations used for building neural networks like loss functions, activation functions, and <a href="/resource/pavq7noze2">convolution</a> operations.
            </td>
        </tr>
        <tr>
            <td>
                torch.optim
            </td>
            <td>
                A subpackage that contains standard optimization operations like SGD and Adam.
            </td>
        </tr>
        <tr>
            <td>
                torch.utils
            </td>
            <td>
                A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier.
            </td>
        </tr>
        <tr>
            <td>
                torchvision
            </td>
            <td>
                A package that provides access to popular datasets, model architectures, and image transformations for computer vision.
            </td>
        </tr>
    </tbody>
</table>


Pytorch uses Dynamic Computational Grpahs unlike Static Computational Graphs, the graph can be built on the fly and this makes things way easier.

## Why GPUs

- Parallel Computation
- GPUs are more suited for parallel computation
- Deep Learning takes advanatge of GPUs
- Neural Networks are `Embarassingly Parallel`

## CUDA and GPU Computation

A software layer that provides API for developers to develop GPU-accelerated applications ( developed by NVDIA )

Taking advanatage of CUDA is extremly advantageous with pytorch

```Python
t = torch.tensor([1,2,3])

t = t.cuda() # t is now on the GPU
```

Tensor objects by default are on CPU, and all operations are done on the CPU

Q) Why not run all computations on GPU?

A) Moving data to GPU is a costly operations. Use GPUs only when the task can be done in parallel.

**GPU Computing Stack** :

**Top Layer** : Frameworks, Apps ( eg: Pytorch )

**Middle Layer** : GPU API Software ( eg: CUDA ) and Libraries( eg: CuDNN )

**Bottom Layer** : GPU Hardware ( eg: NVIDIA )

# Tensors 

---


Primary Data structures used by PyTorch are Tensors

### Terms and Definitions

**Rank of a Tensor** : The number of Dimentions of a Tensor. Tells us how many indicies are required access a specific element of a Tensor.

**Axes of a Tensor** : Element runs along the axes.  

The length of an axis is the number of indices along that axis.

**Shape of Tensor** : 

The shape of a tensor is determined by the length of each axis, so if we know the shape of a given tensor, then we know the length of each axis, and this tells us how many indexes are available along each axis.

```Python
t.shape
```


#### Tensor Reshape

```Python
t.reshape(m,n)
```

Make sure the reshape is legal by making sure that the total number of elements are same after the reshape.

#### Input shape and Feature Maps

For an image Dataset 

`[N,C,H,W]` 

N : Number of Images ( Batch Size )

C : Number of channels

H : Height of the Image

W : Width of the Image

### PyTorch Tensors Explained

**Creating Tensors**

```Python
t = torch.Tensor() # Creating a Tensor with no data

data = np.array([1,2,3]) # Numpy array

t = torch.Tensor(data) # Creating a Tensor from Numpy array

t = torch.tensor(data) # Factory Menthod

t = torch.as_tensor(data)

t = torch.from_numpy(data)
```

**Creating Special tensors**

```Python
torch.eye(2) # Identity Matrix, number of rows are taken as argument

torch.zeros(2,2) # Zero Matrix

torch.ones(2,2) # Ones Matrix

torch.rand(2, 2) # Random Matrix
```

**Tensor Attributes** 

```Python
t.dtype # The dtype of the data in Tensor

t.device # The device on which the tensor is stored and computations are performed

t.layout # How the data is stored in memory
```

**Note** :

* Tensors contains data of a uniform type
* Tensor operations are performed between tensors that are of same dtype and are on the same device