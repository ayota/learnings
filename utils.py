"""
Scope decorators and NN helper functions.

"""
import functools
import tensorflow as tf

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def weight_variable(shape):
    """
    Function to intialize weights for CNN. Initialized with positive noise from
    normal distribution to prevent "dead neurons" (when neuron gets stuck in
    perpetually inactive state) and to also break symmetry and prevent 0 gradients.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Function to initialize bias.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    No frills convolution operation. Stride = 1 and add zero-padding so output is same
    size as input.

    Stride controls how the filter moves across the input volume.

    Zero-padding allows the conservation of the shape of the input, because dimensions
    decrease as filter makes its way across the input.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    Max pooling over 2x2 blocks. This moves across the layer and takes the max of
    4 blocks and puts in (1,1) spot, then repeats for whole input, and zero-pads
    to maintain shape.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
