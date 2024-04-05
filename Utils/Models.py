import os
from scipy import ndimage
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# 设置默认字体为黑体
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']  # 黑体
# 或者设置能够支持中文的其他字体名称
# plt.rcParams['font.family'] = ['SimSun']  # 宋体

# 对于负号显示问题，确保正常显示
plt.rcParams['axes.unicode_minus'] = False


class MyModels(object):
    def __init__(self, map_length=25, zmax=100, x=0, y=0, z=1.5) -> None:
        self.map_length = map_length
        self.zmax = zmax
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def gradient(F):
        fx = np.gradient(F, axis=0)
        fy = np.gradient(F, axis=1)
        grad = np.sqrt(fx ** 2 + fy ** 2)
        return grad


class Ellipse(MyModels):
    miu_0 = 4 * np.pi * (1e-7)  # μ0为真空磁导率
    pi = np.pi

    def __init__(self, map_length=25, zmax=100, a=0.4, b=0.1, c=0.1,
                 gama=90.0, theta=0.0, phi=-5.0, x=0, y=0, z=1.5, b_0=55000.0, I=70.0, D=3.5) -> None:

        self.map_length = map_length
        self.zmax = zmax
        self.X = np.linspace(-map_length, map_length, zmax)  # 0,1,...x
        self.Y = np.linspace(-map_length, map_length, zmax)
        self.center = [x, y, z]  # 中心点坐标
        self.axis = [a, b, c]  # 三个轴长 a> b > c
        self.D = np.radians(D)  # 磁偏角
        self.Latitude = I
        self.I = np.radians(I)  # 磁倾角
        # self.I = np.arctan(2*np.tan(np.radians(I)))  # 磁倾角
        self.b_0 = b_0
        self.B_0 = self.__dicichang()

        if b == c:
            self.axis = [a, b, b]
            self.e = a/b  # 横纵轴比
        # 椭球的方向
        self.gama = gama
        self.theta = theta
        self.phi = phi
        # 转为rad值
        self.gama_rad = np.radians(self.gama)
        self.theta_rad = np.radians(self.theta)
        self.phi_rad = np.radians(self.phi)

        self.V = self.__volme()  # 体积
        self.A = self.__euler_angles()  # 欧拉旋转角
        self.X_d = self.__effective_permeability_matrix()  # 计算有效磁化率矩阵
        self.m_i = self.__total_magnetic_dipole_moment()  # 总磁偶极矩
        self.F = self.__ji_suan_yi_chang_vectorized()

    # 计算体积

    def __volme(self):
        return 4/3*self.pi*self.axis[0]*self.axis[1]*self.axis[2]

    # 计算地磁场
    def __dicichang(self):
        b0 = self.b_0 * np.array([np.cos(self.I)*np.cos(self.D),
                                 np.cos(self.I)*np.sin(self.D), np.sin(self.I)])
        return b0
    # 计算欧拉旋转角

    def __euler_angles(self):
        A = np.array([
            [np.cos(self.gama_rad) * np.cos(self.phi_rad), -np.cos(self.gama_rad)
             * np.sin(self.phi_rad), -np.sin(self.gama_rad)],
            [np.sin(self.theta_rad) * np.sin(self.gama_rad) * np.cos(self.phi_rad) + np.cos(self.theta_rad) * np.sin(self.phi_rad),
             -np.sin(self.theta_rad) * np.sin(self.gama_rad) *
             np.sin(self.phi_rad) + np.cos(self.theta_rad) *
             np.sin(self.phi_rad),
             np.sin(self.theta_rad) * np.cos(self.gama_rad)],
            [np.cos(self.theta_rad) * np.sin(self.gama_rad) * np.cos(self.phi_rad) - np.sin(self.theta_rad) * np.sin(self.phi_rad),
             -np.cos(self.theta_rad) * np.sin(self.gama_rad) *
             np.sin(self.phi_rad) - np.sin(self.theta_rad) *
             np.cos(self.gama_rad),
             np.cos(self.theta_rad) * np.cos(self.gama_rad)]
        ])
        return A

    # 计算有效磁化率矩阵
    def __effective_permeability_matrix(self):
        E = np.log(self.e - np.sqrt(self.e**2 - 1)) / np.sqrt(self.e**2 - 1)
        alpha_1 = (self.e * (self.e + E)) / (self.e**2 - 1)
        alpha_3 = (-2 * self.e * (self.e**(-1) + E)) / (self.e**2 - 1)
        alpha_2 = (self.e * (self.e + E)) / (self.e**2 - 1)
        v = [2/alpha_1, 2/alpha_2, 2/alpha_3]
        X_d = np.diag(v)
        return X_d

    # 计算总磁偶极矩

    def __total_magnetic_dipole_moment(self):
        try:
            m_i = (self.V / self.miu_0) * \
                self.A.T @ self.X_d @ self.A @ self.B_0
        except Exception as e:
            # 异常处理
            raise RuntimeError(
                "Failed to calculate total magnetic dipole moment: {}".format(e))
        return m_i

    def __ji_suan_yi_chang(self):
        X = self.X
        Y = self.Y
        miu_0 = self.miu_0
        m_i = self.m_i
        I = self.I
        D = self.D
        Z = 0
        x_0, y_0, z_0 = self.center
        X_0, Y_0 = np.meshgrid(X, Y)
        igrf = np.array([np.cos(I)*np.cos(D), np.cos(I) *
                        np.sin(D), np.sin(I)], dtype=float)
        self.igrf = igrf
        temp1 = np.empty_like(X_0, dtype=float)
        temp2 = np.empty_like(X_0, dtype=float)
        temp3 = np.empty_like(X_0, dtype=float)
        temp4 = np.empty_like(X_0, dtype=object)
        temp5 = np.empty_like(X_0, dtype=object)
        b = np.empty_like(X_0, dtype=object)
        F1 = np.empty_like(X_0, dtype=float)
        for i in range(X_0.shape[0]):
            for j in range(Y_0.shape[1]):
                R = np.array([X_0[i, j] - x_0, Y_0[i, j] -
                             y_0, Z - z_0], dtype=float)
                RR = np.linalg.norm(R)
                temp1[i, j] = miu_0 / (4 * np.pi * RR ** 3)
                temp2[i, j] = 3 / RR ** 2
                temp3[i, j] = np.dot(R, m_i)
                temp4[i, j] = temp3[i, j] * R
                temp5[i, j] = temp2[i, j] * temp4[i, j] - m_i
                b[i, j] = temp1[i, j] * temp5[i, j]

                F1[i, j] = b[i, j] @ igrf.T

        return F1

    def __ji_suan_yi_chang_vectorized(self):
        miu_0 = self.miu_0
        m_i = self.m_i
        x_0, y_0, z_0 = self.center
        X_0, Y_0 = np.meshgrid(self.X, self.Y)
        Z = np.zeros_like(X_0)

        R = np.stack((X_0 - x_0, Y_0 - y_0, Z - z_0), axis=-1)
        RR = np.linalg.norm(R, axis=-1)
        temp1 = miu_0 / (4 * np.pi * RR ** 3)
        temp2 = 3 / RR ** 2
        temp3 = np.tensordot(R, m_i, axes=(-1, 0))
        temp4 = temp3[..., None] * R
        temp5 = temp2[..., None] * temp4 - m_i
        b = temp1[..., None] * temp5
        I = self.I
        D = self.D
        igrf = np.array([np.cos(I)*np.cos(D), np.cos(I) *
                        np.sin(D), np.sin(I)], dtype=float)
        self.igrf = igrf
        F1 = np.tensordot(b, igrf, axes=(-1, -1))

        return F1

    def parameter(self):
        """
        返回参数 x y z r e I
        """
        pass

    def Plot_X_array_raw(self, levels=16, cmap='rainbow'):
        map_length = self.map_length
        zmax = self.zmax

        X = self.X
        Y = self.Y

        plt.contourf(X, Y, self.F, levels=levels, cmap=cmap)
        plt.xlabel('位置X (m)')
        plt.ylabel('位置Y (m)')
        plt.xticks(np.arange(-map_length, map_length, step=5))
        plt.yticks(np.arange(-map_length, map_length, step=5))

        plt.title(f'纬度 = {self.Latitude}°')

        clb = plt.colorbar()
        clb.set_label('磁通密度 (nT)', labelpad=15, y=0.5, rotation=270)

        plt.show()

    def YOLO_box(self):
        self.G = self.gradient(self.F)
        self.GG = self.gradient(self.G)


class Dipole(MyModels):
    miu_0 = 4 * np.pi * (1e-7)  # μ0为真空磁导率
    Br = 47000
    pi = np.pi
    ksi = 0.1

    def __init__(self, map_length=25, zmax=100, x=0, y=0, h=1, r=0.1, Latitudes=70,
                 H_capteur_bas=0, h_capteur_haut=1):
        self.map_length = map_length
        self.zmax = zmax
        self.x = x
        self.y = y
        self.h = h
        self.r = r
        self.V = np.round(((4)*(self.pi)*(self.r**3)) / 3, decimals=3)
        self.Latitudes = Latitudes
        self.I = np.round(Latitudes, decimals=2)
        self.H_capteur_bas = H_capteur_bas
        self.H_capteur_haut = h_capteur_haut
        self.m = self.Br * self.ksi * self.V

        self.F = self.Anomalie()
        self.bbox = self.YOLO_box()

    @staticmethod
    def grid(zmax, map_length):
        X = np.linspace(-map_length, map_length, zmax).reshape(zmax, 1)
        Y = np.linspace(-map_length, map_length, zmax).reshape(1, zmax)
        return X, Y

    # def calculate_HX_ZZ_TT(self):
    #     pass
    #     u0=4*pi*10^-7;
    #     h=1;    %%深度
    #     r=0.5;        %%%半径
    #     V=(4/3)*pi*r^3;
    #     M=10;       %%%磁化强度
    #     I0=0.5*pi/4;   %%倾斜磁化
    #     I1=pi/2;    %%%垂直磁化
    #     A=pi/2;     %%磁偏角
    #     m=M*V;
    #     term1 = (2*x**2 - y**2 - h**2) * np.cos(I0) * np.sin(A) + \
    #         3*x*y*np.cos(I0)*np.sin(A) - 3*x*h*np.sin(I0)
    #     HX = u0 * m * term1 / (4*np.pi*(x**2 + y**2 + h**2)**(5/2))

    #     term2 = (2*h**2 - x**2 - y**2) * np.sin(I1) - 3*x*h * \
    #         np.cos(I1)*np.sin(A) - 3*y*h*np.cos(I1)*np.sin(A)
    #     ZZ1 = u0 * m * term2 / (4*np.pi*(x**2 + y**2 + h**2)**(5/2))

    #     term3 = (2*h**2 - x**2 - y**2) * np.sin(I0) - 3*x*h * \
    #         np.cos(I0)*np.sin(A) - 3*y*h*np.cos(I0)*np.sin(A)
    #     ZZ = u0 * m * term3 / (4*np.pi*(x**2 + y**2 + h**2)**(5/2))

    #     term4 = (2*h**2 - x**2 - y**2) * np.sin(I0)**2 + (2*x**2 - y**2 - h**2) * np.cos(I0)**2 * np.cos(A)**2 + \
    #             (2*y**2 - x**2 - h**2) * np.cos(I0)**2 * np.sin(A)**2 - 3*x*h*np.sin(2*I0)*np.cos(A) + \
    #         3*x*y*np.cos(I0)**2*np.sin(2*A) - 3*y*h*np.sin(2*I0)*np.sin(A)
    #     TT = u0 * m * term4 / (4*np.pi*(x**2 + y**2 + h**2)**(5/2))

    #     return TT

    def Anomalie(self):
        zmax = self.zmax
        map_length = self.map_length
        H_capteur_bas = self.H_capteur_bas
        H_capteur_haut = self.H_capteur_haut
        I = self.I
        x = self.x
        y = self.y
        h = self.h
        m = self.m
        X, Y = self.grid(zmax, map_length)
        # Array with magnetic induction values
        X_array_raw = np.zeros((zmax, zmax))
        r_bas = np.sqrt(
            (X-x)**2+(Y-y)**2+(h + H_capteur_bas)**2)

        r_haut = np.sqrt(
            (X-x)**2+(Y-y)**2+(h + H_capteur_haut)**2)

        h_x_y_bas = 2*(((h+H_capteur_bas)**2) - (X-x)
                       ** 2 - (Y-y)**2) * np.sin(I)

        h_x_y_haut = 2*(((h+H_capteur_haut)**2) - (X-x)
                        ** 2 - (Y-y)**2) * np.sin(I)

        y_z_cos = np.cos(I) * 3*np.outer(h, Y-y)

        Anomalie_bas = np.divide((h_x_y_bas - y_z_cos), r_bas**5)

        Anomalie_haut = np.divide((h_x_y_haut - y_z_cos), r_haut**5)

        X_array_raw[:, :] = np.outer(
            m, Anomalie_bas-Anomalie_haut).reshape(zmax, zmax)
        return X_array_raw

    def parameter(self):
        """
        返回参数 x y z r e I
        """
        pass

    def Plot_X_array_raw(self, levels=16, cmap='seismic'):
        map_length = self.map_length
        zmax = self.zmax

        X = np.linspace(-map_length, map_length, zmax)
        Y = np.linspace(-map_length, map_length, zmax)

        # 调用 .ravel() 方法后，它会返回一个新的视图，该视图是一维的，
        # 并且保持了原数组的所有元素顺序，但不改变底层数据。这样，
        # 你就可以通过一个简单的索引来遍历所有的子图，而无需
        # 关心它们在原始网格中的具体位置，这对于循环访问所有子图并进行统一操作非常方便。
        # 如下便利
        plt.contourf(X, Y, self.F, levels=levels, cmap=cmap)
        plt.xlabel('位置X (m)')
        plt.ylabel('位置Y (m)')
        plt.xticks(np.arange(-map_length, map_length, step=5))
        plt.yticks(np.arange(-map_length, map_length, step=5))

        plt.title(f'纬度 = {self.Latitudes}°')

        clb = plt.colorbar()
        clb.set_label('磁通密度 (nT)', labelpad=15, y=0.5, rotation=270)

        plt.show()

    def YOLO_box(self):
        """
        根据给定的参数生成一个包含多个异常的边界框。

        参数：
        N_latitudes (int): 纬度的个数。
        n_examples (int): 异常的个数。
        h_array (ndarray): 每个纬度上异常的高度数组。
        map_length (float): 地图的长度。
        zmax (float): 深度的最大值。

        返回：
        bbox (ndarray): 形状为(N_latitudes, n_examples, 4)的边界框数组。
        """

        # 这个函数要求深度的最小值为1米
        zmax, map_length = self.zmax, self.map_length
        real_to_pixel = zmax / (map_length * 2)
        real_to_pixel = 1

        if self.h < 1:
            raise RuntimeError('深度的最小值为1米')

        bbox = np.zeros((4))
        y0, x0 = self.x*real_to_pixel, self.y*real_to_pixel

        lat_i = self.Latitudes
        if 0 <= lat_i < 15:
            # print('if 1')
            w_base = 3.7 * real_to_pixel
            h_base = 5.4 * real_to_pixel
            w_in = 0.5 * real_to_pixel
            h_in = 0.67 * real_to_pixel
        elif 15 <= lat_i < 45:
            # print('elif 1')
            w_base = 4.6 * real_to_pixel
            h_base = 5.2 * real_to_pixel
            w_in = 0.5 * real_to_pixel
            h_in = 0.8 * real_to_pixel
        elif 45 <= lat_i < 75:
            # print('elif 2')
            w_base = 3.4 * real_to_pixel
            h_base = 4.2 * real_to_pixel
            w_in = 0.5 * real_to_pixel
            h_in = 0.65 * real_to_pixel
        else:
            # print('else')
            w_base = 3.6 * real_to_pixel
            h_base = 3.6 * real_to_pixel
            w_in = 0.40 * real_to_pixel
            h_in = 0.40 * real_to_pixel
        n = int((self.h - 0.8)/0.2)
        n = 1
        t1 = n*w_in
        t2 = n*h_in

        bbox[:] = np.array(
            [x0-(h_base/2)-t2/2, y0-(w_base/2)-t1/2, h_base+t1, w_base+t2])
        #   x                       y                w               h

        # 边界处理
        b0, b1, b2, b3 = bbox
        x2 = b0 + b2
        y2 = b1 + b3
        bbox[0] = max(-map_length, min(map_length, bbox[0]))
        bbox[1] = max(-map_length, min(map_length, bbox[1]))
        x2 = max(-map_length, min(map_length, x2))
        y2 = max(-map_length, min(map_length, y2))
        bbox[2] = x2 - bbox[0]
        bbox[3] = y2 - bbox[1]

        return bbox

    def plt_with_box(self, levels=16, cmap='seismic'):
        map_length = self.map_length
        zmax = self.zmax

        X = np.linspace(-map_length, map_length, zmax)
        Y = np.linspace(-map_length, map_length, zmax)
        box_orientation = self.bbox
        x0 = box_orientation[0]
        y0 = box_orientation[1]
        bwidth = box_orientation[2]
        bheight = box_orientation[3]
        rect_real = Rectangle((x0, y0), bwidth, bheight,
                              edgecolor='r', facecolor="none")
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.contourf(X, Y, self.F, levels=levels, cmap=cmap)
        ax.set_xlabel('位置X (m)')
        ax.set_ylabel('位置Y (m)')
        ax.add_patch(rect_real)
        ax.set_xticks(np.arange(-map_length, map_length, step=5))
        ax.set_yticks(np.arange(-map_length, map_length, step=5))

        ax.set_title(f'纬度 = {self.Latitudes}°')


kind = [Dipole, Ellipse]


def generate_parameters_of_Dipole(
        r_max=0.18, h_max=1.8,
        data_num=4096, Rfrequency=0.02, Hfrequency=0.2, nummax=9, seed=56):
    """ 
    返回一个Parameters_array,包含h倍率,r倍率的所有排列组合
    Parameters_array[i,:]为第i个
    """
    r_min = 0.1
    r_max = r_max
    r_array = np.arange(r_min, r_max, Rfrequency)
    h_min = 1
    h_max = h_max
    h_array = np.arange(h_min, h_max, Hfrequency)

    n_examples = r_array.shape[0] * h_array.shape[0]

    Parameters_array = np.zeros((n_examples, 2))

    i_p = 0

    for i_r in r_array:
        for i_h in h_array:
            Parameters_array[i_p, 0] = i_r
            Parameters_array[i_p, 1] = i_h
            i_p += 1
    num_of_rh = Parameters_array.shape[0]
    np.random.seed(seed)
    n = np.random.randint(num_of_rh, size=(data_num, nummax))
    Parameters_array_of_all = np.zeros((data_num, nummax, 2))
    for i in range(data_num):
        for j in range(nummax):
            Parameters_array_of_all[i, j, :] = Parameters_array[n[i, j]]

    return Parameters_array_of_all


def generate_parameters_of_(
        r_max=0.18, h_max=1.8,
        data_num=4096, Rfrequency=0.02, Hfrequency=0.2, nummax=9, seed=56):
    """ 
    返回一个Parameters_array,包含h倍率,r倍率的所有排列组合
    Parameters_array[i,:]为第i个
    """
    r_min = 0.1
    r_max = r_max
    r_array = np.arange(r_min, r_max, Rfrequency)
    h_min = 1
    h_max = h_max
    h_array = np.arange(h_min, h_max, Hfrequency)

    n_examples = r_array.shape[0] * h_array.shape[0]

    Parameters_array = np.zeros((n_examples, 2))

    i_p = 0

    for i_r in r_array:
        for i_h in h_array:
            Parameters_array[i_p, 0] = i_r
            Parameters_array[i_p, 1] = i_h
            i_p += 1
    num_of_rh = Parameters_array.shape[0]
    np.random.seed(seed)
    n = np.random.randint(num_of_rh, size=(data_num, nummax))
    Parameters_array_of_all = np.zeros((data_num, nummax, 2))
    for i in range(data_num):
        for j in range(nummax):
            Parameters_array_of_all[i, j, :] = Parameters_array[n[i, j]]

    return Parameters_array_of_all


def generate_random_coordinate_Dipole(
        map_length=25, rmax_grid=0.2, data_num=4096, nummax=9, rondom_seed=43):
    np.random.seed(rondom_seed)
    L = rmax_grid * 2
    X = np.arange(-map_length, map_length, L)
    coordinate = np.random.choice(X, size=(data_num, nummax, 2))
    return coordinate


def generate_random_muti_dipole(
        r_max=0.18, h_max=1.8, Hfrequency=0.2, Rfrequency=0.02,
        map_length=25, r_max_grid=0.2,
        nummax=9, data_num=4096, seed=43, zmax=100):
    """
    返回parameters, num_of_dipoles, data
    """
    global kind
    np.random.seed(seed)
    datas = np.zeros([data_num, zmax, zmax])
    parameters_n_of_hr = generate_parameters_of_Dipole(
        r_max, h_max, data_num, Rfrequency, Hfrequency, nummax)  # (i,2) h,r
    parameters_n_of_xy = generate_random_coordinate_Dipole(
        map_length, r_max_grid, data_num, nummax)  # (data_num,9,2)
    num_of_dipoles = np.random.randint(8, size=data_num) + 1

    parameter = np.concatenate(
        (parameters_n_of_xy, parameters_n_of_hr), axis=2)  # xyhr
    parameter = np.round(parameter, 2)
    bbox = np.zeros((data_num, nummax, 4))
    for i in range(data_num):
        for j in range(num_of_dipoles[i]):
            x, y, r, h = parameter[i, j]
            tem_data = Dipole(map_length, zmax, x, y, h, r)
            ano = tem_data.F
            bbox[i, j] = tem_data.YOLO_box()
            datas[i] = datas[i] + ano

    return parameter, num_of_dipoles, bbox, datas


def generate_random_muti_mix_data(
        r_max=0.18, h_max=1.8, Hfrequency=0.2, Rfrequency=0.02,
        map_length=25, r_max_grid=0.2,
        nummax=9, data_num=4096, seed=43, zmax=100):
    """
    返回parameters, num_of_dipoles, data
    """
    global kind
    np.random.seed(seed)
    datas = np.zeros([data_num, zmax, zmax])
    # 生成坐标参数
    parameters_n_of_xy = generate_random_coordinate_Dipole(
        map_length, r_max_grid, data_num, nummax)  # (data_num,9,2)
    # 生成球体参数
    parameters_n_of_hr = generate_parameters_of_Dipole(
        r_max, h_max, data_num, Rfrequency, Hfrequency, nummax)  # (i,2) h,r
    # 生成
    num_of_dipoles = np.random.randint(8, size=data_num) + 1

    parameter = np.concatenate(
        (parameters_n_of_xy, parameters_n_of_hr), axis=2)  # xyhr
    parameter = np.round(parameter, 2)
    bbox = np.zeros((data_num, nummax, 4))
    for i in range(data_num):
        for j in range(num_of_dipoles[i]):
            x, y, r, h = parameter[i, j]
            tem_data = Dipole(map_length, zmax, x, y, h, r)
            ano = tem_data.F
            bbox[i, j] = tem_data.YOLO_box()
            datas[i] = datas[i] + ano

    return parameter, num_of_dipoles, bbox, datas


def X_array(datas, map_lenght=25, new_size=416):
    zmax = datas.shape[1]
    pixel_to_real = ((map_lenght*2)/zmax)
    real_to_pixel = zmax/(map_lenght*2)
    conversion = (2*map_lenght)/zmax
    yolo_conversion = (new_size/zmax)
    No_pad_left = int((zmax) - zmax/2)
    No_pad_right = int((zmax) + zmax/2)
    ##########################

    ##########################
    n_examples = datas.shape[0]
    X_data = np.zeros((n_examples, new_size, new_size))

    border_o = int((zmax)/2)

    # data变换大小
    for i_n_examples in range(n_examples):
        img = datas[i_n_examples, :, :]
        X_data_padded = cv2.copyMakeBorder(
            img, border_o, border_o, border_o, border_o, 0)
        X_array_os = X_data_padded[No_pad_left:No_pad_right,
                                   No_pad_left:No_pad_right]
        X_array_yolo = cv2.resize(
            X_array_os, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
        X_data[i_n_examples, :, :] = X_array_yolo
        ##########################

    # 变换bbox

    return X_data


def Plot_X_data(num_of_dipoles, bbox, datas, map_lenght=25, num=2):

    n_examples = datas.shape[0]
    zmax = datas.shape[1]

    pixel_to_real = ((map_lenght*2)/zmax)
    real_to_pixel = zmax/(map_lenght*2)

    X = np.linspace(-map_lenght, map_lenght, zmax)
    Y = np.linspace(-map_lenght, map_lenght, zmax)

    rows, cols = 1, num
    height_2 = 14
    width_2 = 8
    fig, axs = plt.subplots(rows, cols, figsize=(height_2, width_2))
    # fig.subplots_adjust(hspace = 0, wspace=0)

    axs = axs.ravel()
    examples = np.random.randint(n_examples, size=num)

    for lat_i in range(num):
        rect_real = []
        rect_pixel = []
        example = examples[lat_i]
        for n_ii in range(num_of_dipoles[example]):

            x0, y0, bwidth, bheight = (bbox[example, n_ii]) * real_to_pixel
            rect_pixel.append(Rectangle((x0, y0), bwidth,
                                        bheight, edgecolor='r', facecolor="none"))

            x0_2, y0_2, bwidth_2, bheight_2 = (bbox[example, n_ii])

            print(
                f"Dipole_{n_ii}: x0: {np.round(x0_2,1)}, y0: {np.round(y0_2,1)}, w: {round(bwidth_2,1)}, h:{round(bheight_2,1)}")

            rect_real.append(Rectangle((x0_2, y0_2), bwidth_2,
                                       bheight_2, edgecolor='r', facecolor="none"))

        axs[lat_i].contourf(X, Y, datas[example, :, :],
                            levels=18, cmap='seismic')
        axs[lat_i].set_xlabel('Position X (m)')
        axs[lat_i].set_ylabel('Position Y (m)')
        axs[lat_i].set_xticks(np.arange(-25, 25, step=2))
        axs[lat_i].set_yticks(np.arange(-25, 25, step=2))
        for n_iii in range(num_of_dipoles[example]):
            axs[lat_i].add_patch(rect_real[n_iii])

        title = 'test'
        axs[lat_i].set_title(f'Latitude = {title}°')

    plt.tight_layout()


def add_gaussian_noise(X_data_array, mean=0, var=0.1, n_models_with_noise=100, seed=76):
    sigma = var ** 0.5
    zmax = X_data_array.shape[1]
    nummax = X_data_array.shape[0]
    np.random.seed(seed)

    add_noise_id = np.random.choice(
        nummax, size=n_models_with_noise, replace=False)

    for i_noise_models in range(n_models_with_noise):

        m = add_noise_id[i_noise_models]

        if (0 < np.max(X_data_array[m, :, :])) and (np.max(X_data_array[m, :, :]) < 10):

            var = 0.01

        elif (-10 < np.min(X_data_array[m, :, :])) and (np.min(X_data_array[m, :, :]) < 0):

            var = 0.01

        elif (10 < np.max(X_data_array[m, :, :])) and (np.max(X_data_array[m, :, :]) < 100):

            var = 1

        elif (-100 < np.min(X_data_array[m, :, :])) and (np.min(X_data_array[m, :, :]) < -10):

            var = 1

        elif (100 < np.max(X_data_array[m, :, :])) and (np.max(X_data_array[m, :, :]) < 1000):

            var = 5

        elif (-1000 < np.min(X_data_array[m, :, :])) and (np.min(X_data_array[m, :, :]) < -100):

            var = 5

        else:
            var = 10

        sigma = var ** 0.5

        gaussian = np.random.normal(mean, sigma, (zmax, zmax))
        X_data_array[m] = X_data_array[m] + gaussian
    return add_noise_id


def Plot_X_data_with_noise(num_of_dipoles, bbox, datas, with_noise_id, map_lenght=25, num=2):

    n_examples = with_noise_id.shape[0]
    zmax = datas.shape[1]

    pixel_to_real = ((map_lenght*2)/zmax)
    real_to_pixel = zmax/(map_lenght*2)

    X = np.linspace(-map_lenght, map_lenght, zmax)
    Y = np.linspace(-map_lenght, map_lenght, zmax)

    rows, cols = 1, num
    height_2 = 14
    width_2 = 8
    fig, axs = plt.subplots(rows, cols, figsize=(height_2, width_2))
    # fig.subplots_adjust(hspace = 0, wspace=0)

    axs = axs.ravel()
    examples = np.random.choice(n_examples, size=num, replace=False)

    for lat_i in range(num):
        rect_real = []
        rect_pixel = []
        example = with_noise_id[examples[lat_i]]
        for n_ii in range(num_of_dipoles[example]):

            x0, y0, bwidth, bheight = (bbox[example, n_ii]) * real_to_pixel
            rect_pixel.append(Rectangle((x0, y0), bwidth,
                                        bheight, edgecolor='r', facecolor="none"))

            x0_2, y0_2, bwidth_2, bheight_2 = (bbox[example, n_ii])

            print(
                f"Dipole_{n_ii}: x0: {np.round(x0_2,1)}, y0: {np.round(y0_2,1)}, w: {round(bwidth_2,1)}, h:{round(bheight_2,1)}")

            rect_real.append(Rectangle((x0_2, y0_2), bwidth_2,
                                       bheight_2, edgecolor='r', facecolor="none"))

        test = axs[lat_i].contourf(X, Y, datas[example, :, :],
                                   levels=50, cmap='gray')
        plt.colorbar(test, ax=axs[lat_i])
        axs[lat_i].set_xlabel('Position X (m)')
        axs[lat_i].set_ylabel('Position Y (m)')
        axs[lat_i].set_xticks(np.arange(-25, 25, step=2))
        axs[lat_i].set_yticks(np.arange(-25, 25, step=2))
        for n_iii in range(num_of_dipoles[example]):
            axs[lat_i].add_patch(rect_real[n_iii])

        title = 'test'
        axs[lat_i].set_title(f'Latitude = {title}°')

    plt.tight_layout()


new_size = 416
IMG_size = (new_size, new_size)


def convert_to_YOLO(
        num_of_dipoles, bbox, X_data_array, root_dir=None, map_length=25):
    num_of_datas = X_data_array.shape[0]
    X_array_data = X_data_array.reshape(
        X_data_array.shape[0], X_data_array.shape[1], X_data_array.shape[2], 1)
    length = int(map_length*2)
    bbox[:, :, 0] = bbox[:, :, 0]+bbox[:, :, 2]/2 + map_length
    bbox[:, :, 1] = bbox[:, :, 1]+bbox[:, :, 3]/2 + map_length

    YOLO_box = bbox/length
    if root_dir is None:
        root_dir = '/home/jiajianhao/文档/cnn/my_Magnetic_methods-master/data/YOLOv8'  # type: ignore
    n1 = int(num_of_datas*0.7)
    n2 = int(num_of_datas*0.95)
    for bb_n_models in range(num_of_datas):
        if bb_n_models < n1:
            flag = "train"
            dir_name = f'{root_dir}/{flag}'
            os.chdir(dir_name)
        elif bb_n_models < n2:
            flag = "val"
            dir_name = f'{root_dir}/{flag}'
            os.chdir(dir_name)
        else:
            flag = "test"
            dir_name = f'{root_dir}/{flag}'
            os.chdir(dir_name)
        # Normalize between 0 - 1

        image = (X_array_data[bb_n_models, :, :, :] - np.min(X_array_data[bb_n_models, :, :, :])) / (
            np.max(X_array_data[bb_n_models, :, :, :]) - np.min(X_array_data[bb_n_models, :, :, :]))

        # Converting to rgb

        image_t = np.uint8(image * 255)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_GRAY2RGB)  # type: ignore

        # Saving image

        im = Image.fromarray(image_t)
        im.convert('RGB').save(f"{flag}_{bb_n_models+1}.jpg")

        for bb_n_dipoles in range(num_of_dipoles[bb_n_models]):

            if bb_n_dipoles == 0:

                with open(f"{flag}_{bb_n_models+1}.txt", "w") as f:

                    f.write('0' + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 0]) + ' ' + str(
                        YOLO_box[bb_n_models, bb_n_dipoles, 1]) + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 2]) + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 3]))

            else:
                with open(f"{flag}_{bb_n_models+1}.txt", "a") as f:
                    f.write('\n' + '0' + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 0]) + ' ' + str(
                        YOLO_box[bb_n_models, bb_n_dipoles, 1]) + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 2]) + ' ' + str(YOLO_box[bb_n_models, bb_n_dipoles, 3]))


if __name__ == '__main__':
    pass
