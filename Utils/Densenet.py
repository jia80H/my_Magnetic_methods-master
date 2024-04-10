import numpy as np


def cov_to_densenet(
        parameter, num_of_ano, kind_of_data, bbox, datas,
        map_length=25, den_size=64):
    """
    输入:
        parameter: (data_num,nummax,n) n表示n个参数
        num_of_ano: (,data_num) 每个数据点有多少个磁异常
        kind_of_data: (data_num,nummax) 每个数据的类型
        bbox: (data_num,nummax,4) 每个数据点的边界
        datas: (data_num,zmax,zmax) 所有磁异常图
        den_size: 输入到densenet的图像大小
    """
    NofD = parameter.shape[0]
    NofK = parameter.shape[1]
    zmax = datas.shape[1]

    length = int(map_length*2)
    YOLO_box = np.copy(bbox)
    YOLO_box[:, :, 0] = YOLO_box[:, :, 0]+YOLO_box[:, :, 2]/2 + map_length
    YOLO_box[:, :, 1] = YOLO_box[:, :, 1]+YOLO_box[:, :, 3]/2 + map_length
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
            all_densenet[i, j, x2:x3, y2:y3] = datas[i, j, x0:x1, y0:y1]
    return all_densenet


def fliter_acdt_type():
    pass
