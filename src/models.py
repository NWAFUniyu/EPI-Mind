
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import tensorflow as tf
from transfomer import Transformer


def get_model(max_len_en, max_len_pr, nwords, emb_dim):

    enhancers = Input(shape=(max_len_en,))
    promoters = Input(shape=(max_len_pr,))

    embedding_matrix = np.load('embedding_matrix.npy')

    emb_en = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=True)(enhancers)
    emb_pr = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=True)(promoters)

    enhancer_conv_layer = Conv1D(filters=72,#64
                                 kernel_size=36,#40
                                 padding="valid",
                                 activation='relu')(emb_en)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters=72,
                                 kernel_size=36,#
                                 padding="valid",
                                 activation='relu')(emb_pr)
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer)

    transformer1 = Transformer(  encoder_stack=4,
                                feed_forward_size=256,
                                n_heads=8,
                                model_dim=72)

    transformer2 = Transformer(  encoder_stack=4,
                                feed_forward_size=256,
                                n_heads=8,
                                model_dim=72)

    enhancer_trf = transformer1(enhancer_max_pool_layer)
    promoter_trf = transformer2(promoter_max_pool_layer)

    # enhancer_avgpool = GlobalAveragePooling1D()(enhancer_trf)
    # promoter_avgpool = GlobalAveragePooling1D()(promoter_trf)

    enhancer_maxpool = GlobalMaxPooling1D()(enhancer_trf)
    promoter_maxpool = GlobalMaxPooling1D()(promoter_trf)

    # enhancer_pool = tf.concat([enhancer_avgpool, enhancer_maxpool], -1)
    # promoter_pool = tf.concat([promoter_avgpool, promoter_maxpool], -1)

    # merge
    merge = tf.concat([enhancer_maxpool * promoter_maxpool,
                       tf.abs(enhancer_maxpool - promoter_maxpool),
                       ], -1)
    #my
    # bn=BatchNormalization()(merge)
    # dt=Dropout(0.5)(merge)

    merge2 = Dense(50, activation='relu')(merge)
#    merge3 = Dense(20, activation='relu')(merge2)


    preds = Dense(1, activation='sigmoid')(merge2)
    model = Model([enhancers, promoters], preds)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
