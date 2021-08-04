import tensorflow as tf
from tensorflow.keras.layers import Conv1D


class ConvolutionLayer(Conv1D):
    def __init__(self,
                 filters,
                 kernel_size,
                 data_format='channels_last',
                 alpha=100,
                 beta=0.01,
                 bkg_const=[0.25, 0.25, 0.25, 0.25],
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 __name__='ConvolutionLayer',
                 **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.bkg_const = bkg_const
        self.run_value = 1

    def call(self, inputs):
        # print("self.run value is", self.run_value)
        if self.run_value > 2:
            x_tf = self.kernel  # x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])
            # self.alpha = 10
            self.beta = 1/self.alpha
            bkg = tf.constant(self.bkg_const)
            bkg_tf = tf.cast(bkg, tf.float32)

            # print('\n---------------debug---------------\n')
            # print(x_tf[0])
            # print(tf.scalar_mul(100, x_tf[0]))
            # print(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x_tf[0]), axis=1))
            # print(tf.expand_dims(
            #                         tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x_tf[0]), axis=1), axis=1
            #                     ),)
            # print('\n---------------debug---------------\n')
            # filt_list = tf.map_fn(lambda x: tf.math.scalar_mul(self.beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(self.alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis = 1), axis = 1)), tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(self.alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)), tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            filt_list = tf.map_fn(
                lambda x: tf.math.scalar_mul(
                    self.beta,
                    tf.subtract(
                        tf.subtract(
                            tf.subtract(
                                tf.math.scalar_mul(self.alpha, x),
                                tf.expand_dims(
                                    tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1
                                ),
                            ),
                            tf.expand_dims(
                                tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.subtract(
                                                tf.math.scalar_mul(
                                                    self.alpha, x),
                                                tf.expand_dims(
                                                    tf.math.reduce_max(
                                                        tf.math.scalar_mul(self.alpha, x), axis=1
                                                    ),
                                                    axis=1,
                                                ),
                                            )
                                        ),
                                        axis=1,
                                    )
                                ),
                                axis=1,
                            ),
                        ),
                        tf.math.log(
                            tf.reshape(
                                tf.tile(bkg_tf, [tf.shape(x)[0]]),
                                [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                            )
                        ),
                    ),
                ),
                x_tf,
            )

            # filt_list = tf.math.scalar_mul(
            #     self.beta,
            #     tf.subtract(
            #         tf.subtract(
            #             tf.subtract(
            #                 tf.math.scalar_mul(self.alpha, x),
            #                 tf.expand_dims(
            #                     tf.math.reduce_max(
            #                         tf.math.scalar_mul(self.alpha, x),
            #                         axis=1
            #                     ),
            #                     axis=1
            #                 )
            #             ),
            #             tf.expand_dims(
            #                 tf.math.log(
            #                     tf.math.reduce_sum(
            #                         tf.math.exp(
            #                             tf.subtract(
            #                                 tf.math.scalar_mul(self.alpha, x),
            #                                 tf.expand_dims(
            #                                     tf.math.reduce_max(
            #                                         tf.math.scalar_mul(self.alpha, x),
            #                                         axis=1
            #                                     ),
            #                                     axis=1
            #                                 )
            #                             )
            #                         ),
            #                         axis=1
            #                     )
            #                 ),
            #                 axis=1
            #             )
            #         ),
            #         tf.math.log(
            #             tf.reshape(
            #                 tf.tile(
            #                     bkg_tf,
            #                     [ tf.shape(x)[0] ]
            #                 ),
            #                 [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ]
            #             )
            #         )
            #     )
            # )
            # print("type of output from map_fn is", type(filt_list)) ##type of output from map_fn is <class 'tensorflow.python.framework.ops.Tensor'>   shape of output from map_fn is (10, 12, 4)
            #print("shape of output from map_fn is", filt_list.shape)
            # transf = tf.reshape(filt_list, [12, 4, self.filters]) ##12, 4, 512
            transf = tf.transpose(filt_list, [1, 2, 0])
            # type of transf is <class 'tensorflow.python.framework.ops.Tensor'>
            # type of outputs is <class 'tensorflow.python.framework.ops.Tensor'>
            outputs = self._convolution_op(inputs, transf)
        else:
            outputs = self._convolution_op(inputs, self.kernel)

        self.run_value += 1
        return outputs
