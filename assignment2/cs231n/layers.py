from builtins import range
import numpy as np
from numpy import unravel_index


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    num_input = x.shape[0]
    dim = int(np.prod(x.shape)/num_input)
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = x.reshape(num_input, dim).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    x_shape = x.shape
    num_input = x.shape[0]
    dim = int(np.prod(x.shape)/num_input)
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x = x.reshape(num_input, dim)
    dw = np.dot(x.T, dout)
    dx = np.dot(dout, w.T).reshape(x_shape)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dout[np.maximum(0, x) == 0] = 0
    dx = dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var: Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.mean(x, axis = 0) #mini batch mean
        zerocentered_mean = x - sample_mean
        sample_var = np.mean(zerocentered_mean**2, axis = 0)  #mini batch variance
        sample_std = np.sqrt(sample_var + eps)
        inverse_std = np.reciprocal(sample_std)
        normalized_x = (zerocentered_mean) * inverse_std # normalize
        
        out = gamma * normalized_x + beta # scale and shift
        
        #calculate running mean and average
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        #store in cache for the backward path
        cache = (zerocentered_mean, sample_var, sample_std, inverse_std, normalized_x, gamma, eps)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        normalized_x = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma * normalized_x + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #restore cache
    zerocentered_mean, sample_var, sample_std, inverse_std, normalized_x, gamma, eps = cache
    N, D = np.shape(dout)
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    
    dnormalized_x = dout * gamma
    dmean_centered1 = dnormalized_x * inverse_std
    dinverse_std = np.sum(dnormalized_x * zerocentered_mean, axis=0)
    dsample_std = -np.reciprocal(sample_std ** 2) * dinverse_std
    dsample_var = 0.5 * (np.power(sample_var + eps, -0.5)) * dsample_std
    dunnormalized_var = 1./N * np.ones((N, D), dtype=dout.dtype) * dsample_var
    dmean_centered2 = 2 * zerocentered_mean * dunnormalized_var
    dsample_mean = -1 * np.sum(dmean_centered1 + dmean_centered2, axis=0)
    dx1 = dmean_centered1 + dmean_centered2
    dx2 = 1./N  * np.ones((N, D), dtype=dout.dtype) * dsample_mean
    dx = dx1 + dx2
                                  
                                   
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * normalized_x, axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #restore cache, sample_std not required
    zerocentered_mean, sample_var, _, inverse_std, normalized_x, gamma, eps = cache
    N, D = np.shape(dout)
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    # Equations similar to paper. Note: these can be simplified further, 
    # but left like this for clear understanding
    dnormalized_x = dout * gamma
    dsample_var =  np.sum(dnormalized_x * zerocentered_mean, axis=0) \
                      * -0.5 * np.power(sample_var + eps, -3/2)
    dsample_mean = (np.sum(dnormalized_x * (-inverse_std), axis=0)) \
                    + dsample_var * np.sum(-2 * zerocentered_mean, axis=0)/N
    dx = dnormalized_x * inverse_std \
          + dsample_var * 2 * zerocentered_mean/N + dsample_mean/N
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * normalized_x, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    N, C, H, W = x.shape
    F, _, HH, WW = w. shape
    stride = conv_param.get('stride', 0)
    pad = conv_param.get('pad', 1)
    
    assert((H + 2 * pad - HH) % stride == 0) 
    assert((W + 2 * pad - WW) % stride == 0)
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N, F, H_out, W_out))
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    x = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')   
    
    for n in range(N):
        row_start = 0
        for row_out in range(W_out):
            col_start = 0
            for col_out in range(H_out):
                for f in range(F):
                    row_end = row_start + HH
                    col_end = col_start + WW
                    out[n, f, col_out, row_out] = \
                    np.sum(x[n, :, col_start:col_end, row_start:row_end ] * w[f, :, :, :]) + b[f] 
                col_start += stride
            row_start += stride
                
        
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
    N, C, H, W = x.shape
    F, _, HH, WW = w. shape
    
    stride = conv_param.get('stride', 0)    
    pad = conv_param.get('pad', 1)
    
    _, _, H_out, W_out = dout.shape
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    for n in range(N):
        row_start = 0
        for row_out in range(W_out):
            col_start = 0
            for col_out in range(H_out):
                for f in range(F):
                    row_end = row_start + HH
                    col_end = col_start + WW
                    dout_elem = dout[n, f, col_out, row_out]
                    db[f] += dout_elem
                    dx[n, :, col_start:col_end, row_start:row_end ] += dout_elem * w[f, :, :, :]
                    dw[f, :, :, :] += dout_elem * x[n, :, col_start:col_end, row_start:row_end ]
                col_start += stride
            row_start += stride
            
    #ignore gradients for the padded values, reverse operation of padding
    dx = dx[:,:, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 2)    
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 1)
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    
    out = np.zeros((N, C, H_out, W_out))
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    for n in range(N):
        row_start = 0
        for row_out in range(W_out):
            col_start = 0
            for col_out in range(H_out):
                row_end = row_start + pool_height
                col_end = col_start + pool_width
                for c in range(C):
                    out[n, c, col_out, row_out] =\
                    x[n, c, col_start:col_end, row_start:row_end].max()
                col_start += stride
            row_start += stride
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    pool_height = pool_param.get('pool_height', 2)    
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 1)
    N, C, H, W = x.shape
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    
    dx = np.zeros(x.shape)
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    for n in range(N):
        row_start = 0
        for row_out in range(W_out):
            col_start = 0
            for col_out in range(H_out):
                row_end = row_start + pool_height
                col_end = col_start + pool_width
                for c in range(C):
                    max_indexes = (unravel_index(x[n, c, col_start:col_end, \
                                                   row_start:row_end].argmax(), \
                                                 x[n, c, col_start:col_end, row_start:row_end].shape))
                    max_index1 = max_indexes[0] + col_start
                    max_index2 = max_indexes[1] + row_start
                    dout_elem = dout[n, c, col_out, row_out]
                    dx[n, c, max_index1, max_index2] = dout_elem
                col_start += stride
            row_start += stride
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x = np.swapaxes(x, 0, 1)
    out,cache = batchnorm_forward(x.reshape(C, N * H * W).T, gamma, beta, bn_param)
    out = out.T.reshape(C, N, H, W).swapaxes(0,1)
#     gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]
#     beta = beta[np.newaxis, :, np.newaxis, np.newaxis]
#     out, cache = batchnorm_forward(np.mean(x, axis=(2,3)), gamma, beta, bn_param)
#     print(bn_param['running_mean'] )
#     mode = bn_param['mode']
#     eps = bn_param.get('eps', 1e-5)
#     momentum = bn_param.get('momentum', 0.9)

#     N, C, H, W = x.shape
#     running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
#     running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))
    
#     out, cache = None, None

#     ###########################################################################
#     # TODO: Implement the forward pass for spatial batch normalization.       #
#     #                                                                         #
#     # HINT: You can implement spatial batch normalization using the vanilla   #
#     # version of batch normalization defined above. Your implementation should#
#     # be very short; ours is less than five lines.                            #
#     ###########################################################################
#     if mode == 'train':
#         sample_mean = np.mean(x, axis = (0, 2, 3))
#         zerocentered_mean = x - sample_mean[np.newaxis, :, np.newaxis, np.newaxis]
#         sample_var = np.mean(zerocentered_mean**2, axis = (0, 2, 3))
#         sample_std = np.sqrt(sample_var[np.newaxis, :, np.newaxis, np.newaxis] + eps)
#         inverse_std = np.reciprocal(sample_std)
#         normalized_x = (zerocentered_mean) * inverse_std # normalize

#         out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * \
#               normalized_x + beta[np.newaxis, :, np.newaxis, np.newaxis] # scale and shift

#         #calculate running mean and average
#         running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#         running_var = momentum * running_var + (1 - momentum) * sample_var

#         #store in cache for the backward path
#         cache = (zerocentered_mean, sample_var, sample_std, inverse_std, normalized_x, gamma, eps)
    
#     elif mode == 'test':
#         normalized_x = (x - running_mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
#                         np.sqrt(running_var[np.newaxis, :, np.newaxis, np.newaxis] + eps)
#         out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * \
#               normalized_x + beta[np.newaxis, :, np.newaxis, np.newaxis] # scale and shift
    
#     else:
#         raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
#     bn_param['running_mean'] = running_mean
#     bn_param['running_var'] = running_var
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.swapaxes(0,1).reshape(C, N * H * W).T, cache)
    dx = dx.T.reshape(C, N, H, W).swapaxes(0,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
