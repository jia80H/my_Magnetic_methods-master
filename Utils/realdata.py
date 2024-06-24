import numpy as np
import matplotlib.pyplot as plt


def load_box(path):
    data = np.loadtxt(path)
    # print(data)
    m, n = data.shape
    n_of_each = np.zeros((1, 9))
    n_of_each[0] = data.T[0]
    # kind, n_of_each = np.unique(num_of_ano, return_counts=True)
    # print(kind, n_of_each)
    # print(num_of_ano)

    box = data.T[1:]
    box = box.T
    # print(box)
    parameter = np.zeros((1, 9))
    num_of_ano = []
    num_of_ano.append(m)
    bbox = np.zeros((1, m, 4))
    bbox[0] = box
    return parameter, num_of_ano, bbox, n_of_each


def cov_to_densenet_one(
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
