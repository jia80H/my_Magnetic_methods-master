from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MSE, MAE
from tensorflow.keras.initializers import he_normal, random_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Dense, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, Concatenate, SpatialDropout2D, Conv2D, Flatten, BatchNormalization, Dropout
import tensorflow as tf
import seaborn as sns
import cv2 as cv

import numpy as np
import random
import matplotlib.pyplot as plt


def gray_to_rgb(data_array):
    array = data_array.astype(np.float32)
    array = np.expand_dims(array, axis=-1)
    array_rgb = np.zeros(array.shape, dtype=np.float32)
    array_rgb = np.repeat(array_rgb, 3, -1)

    for n_i in range(array.shape[0]):
        array_rgb[n_i] = cv.cvtColor(
            array[n_i], cv.COLOR_GRAY2RGB)

    return array_rgb.astype("float32")


def cov_to_densenet(
        parameter, num_of_ano, bbox, datas,
        map_length=25, den_size=64):
    """
    输入:
        parameter: (data_num,nummax,n) n表示n个参数
        num_of_ano: (,data_num) 每个数据点有多少个磁异常
        bbox: (data_num,nummax,4) 每个数据点的边界,
        datas: (data_num,zmax,zmax) 所有磁异常图
        den_size: 输入到densenet的图像大小
    """
    NofD = parameter.shape[0]
    NofK = parameter.shape[1]
    zmax = datas.shape[-1]
    length = int(map_length*2)
    YOLO_box = np.copy(bbox)
    YOLO_box[:, :, 0] = YOLO_box[:, :, 0] + map_length
    YOLO_box[:, :, 1] = YOLO_box[:, :, 1] + map_length
    # YOLO_box变成索引
    YOLO_box = (YOLO_box/length*zmax).astype(np.uint16)
    index_den = np.zeros_like(YOLO_box, dtype=np.uint16)  # den中xy的索引
    index_den[:, :, 0] = (den_size - YOLO_box[:, :, 2])//2
    index_den[:, :, 1] = (den_size - YOLO_box[:, :, 3])//2
    index_den[:, :, 2] = YOLO_box[:, :, 2] + index_den[:, :, 0]
    index_den[:, :, 3] = YOLO_box[:, :, 3] + index_den[:, :, 1]

    YOLO_box[:, :, 2] = YOLO_box[:, :, 2] + YOLO_box[:, :, 0]
    YOLO_box[:, :, 3] = YOLO_box[:, :, 3] + YOLO_box[:, :, 1]

    all_densenet = np.zeros((NofD, NofK, den_size, den_size))
    for i in range(NofD):
        for j in range(num_of_ano[i]):
            x0, y0, x1, y1 = YOLO_box[i, j]
            x2, y2, x3, y3 = index_den[i, j]
            all_densenet[i, j, y2:y3, x2:x3] = datas[i, y0:y1, x0:x1]

    return all_densenet


def plot_densenet_data(den_data, map_lenght=25):
    zmax = den_data.shape[-1]

    xmin = -map_lenght
    xmax = map_lenght
    ymin = -map_lenght
    ymax = map_lenght

    X = np.linspace(xmin, xmax, zmax)
    Y = np.linspace(ymin, ymax, zmax)

    # Adjust the size of your images
    plt.figure(figsize=(15, 7))

    # Iterate and plot random images 迭代 绘制随机图像
    for i in range(6):

        n_p = int(np.random.randint(0, den_data.shape[0], 1))

        plt.subplot(2, 3, i + 1)
        plt.contourf(X, Y, den_data[n_p, 0, :, :], levels=40, cmap='rainbow')
        plt.axis('on')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        clb = plt.colorbar()
        clb.set_label('磁异常 (nT)',
                      labelpad=15, y=0.5, rotation=270)

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()


def fliter_acdt_type(
        num_of_each, parameter, num_of_ano,
        all_densenet, kind_of_data):
    """
    将输入的数据按照种类划分为不同训练参数
    输入:
        num_of_each: num_of_each_data_in_total = [n1, n2]　utils.models中的变量
        parameter: (data_num,nummax,n) n表示n个参数
        num_of_ano: (,data_num) 每个数据点有多少个磁异常
        all_densenet: (data_num,den_size,den_size) 所有densenet数据
        kind_of_data: (data_num,nummax) 每个数据的类型
        para_n: 返回数据参数中shape的第二维 如(n,9)

    返回:
        all_kind_para:
            [
            [train val test],   一个类型的所有参数   train: [num,n] n个该类型的参数
            [train val test],
            .....
            ]
        all_kind_data:
            [
            [train val test],   一个类型的所有数据   train: [num,den_size,den_size,3] n个该类型的参数
            [train val test],
            .....
            ]

    """

    kind_num = len(num_of_each)
    para_n = parameter.shape[-1]
    zmax = all_densenet.shape[-1]

    all_data = {}  # 未被分割的数据
    all_para = {}

    # 数据初始化
    for k in range(kind_num):
        n = num_of_each[k]  # 该种模型的总数
        all_data[f'{k}'] = np.zeros((n, zmax, zmax))
        all_para[f'{k}'] = np.zeros((n, para_n))

    # 全体数据操作
    count = [0, 0, 0, 0, 0, 0]
    for n in range(all_densenet.shape[0]):
        for nn in range(num_of_ano[n]):
            kind = kind_of_data[n, nn]
            num = count[kind]
            all_data[f'{kind}'][num] = all_densenet[n, nn]
            all_para[f'{kind}'][num] = parameter[n, nn]
            count[kind] += 1

    return all_data, all_para


def split_data(all_data, all_para, n1=0.7, n2=0.95):
    """
    返回训练集、验证集、测试集
    """
    # 数据集分割
    n = all_data.shape[0]  # 该种模型的总数
    n1 = int(n*n1)
    n2 = int(n*n2)
    train_data = all_data[:n1]
    train_para = all_para[:n1]
    val_data = all_data[n1:n2]
    val_para = all_para[n1:n2]
    test_data = all_data[n2:]
    test_para = all_para[n2:]
    return train_data, train_para, val_data, val_para, test_data, test_para


def para_select(start, end, *args):
    paras = []
    for arg in args:
        arg = arg[:, start:end]
        paras.append(arg)

    return paras


def load_densnet():
    pass


def r2(y_true, y_pred):

    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res/(SS_tot + K.epsilon()))

# Dense Block


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
          # Arguments
              x: input tensor 
              stage: index for dense block
              branch: layer index within each dense block
              nb_filter: number of filters
              dropout_rate: dropout rate
              weight_decay: weight decay factor
      '''
    concat_axis = -1

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4

    x = BatchNormalization(epsilon=eps, axis=concat_axis,
                           name=conv_name_base+'_x1_bn')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base +
               '_x1', use_bias=False)(x)

    if dropout_rate:
        x = SpatialDropout2D(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis,
                           name=conv_name_base+'_x2_bn')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = SpatialDropout2D(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
          # Arguments
              x: input tensor
              stage: index for dense block
              nb_filter: number of filters
              compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
              dropout_rate: dropout rate
              weight_decay: weight decay factor
      '''
    eps = 1.1e-5

    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'tanh' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    concat_axis = -1

    x = BatchNormalization(epsilon=eps, axis=concat_axis,
                           name=conv_name_base+'_bn')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1),
               name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = SpatialDropout2D(dropout_rate)(x)

    # x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(
        x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
        weight_decay=1e-4, grow_nb_filters=True):
    ''' 
    Build a dense_block where the output of each conv_block is
    fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to
                append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch,
                       growth_rate, dropout_rate, weight_decay)
        concat_feat = Concatenate(
            axis=-1, name='concat_'+str(stage)+'_'+str(branch))([concat_feat, x])
        # concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
