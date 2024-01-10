import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F
from feeders.manifolds import Manifold
#from utils.math_utils import artanh, tanh

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        # if p_interval == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',
                         align_corners=False).squeeze()  # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def xyz2Spherical(xx, yy, zz):
    r = np.sqrt(pow(xx, 2) + pow(yy, 2) + pow(zz, 2))
    theta = np.arctan(np.sqrt(pow(xx, 2) + pow(yy, 2)) / zz)
    beta = np.arctan(yy / xx)
    return r, theta, beta


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def calcu_curl(data_numpy):
    T, V, C, M = data_numpy.shape  # CTVM
    u = np.zeros([T, V, 3, M])
    omega = np.zeros([T, V, 3, M])
    alpha = np.zeros([T, V, 3, M])
    angular_v = np.zeros([T, V, 3, M])
    angular_a = np.zeros([T, V, 3, M])

    parents = np.array(
        [0, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 7, 12, 11])
    for j in range(T):
        for i in range(1, V):
            joints = data_numpy[j][:, [0, 2, 1]]  # change xyz to xzy of vertical coors  CTVM
            u[j][i] = joints[i] - joints[parents[i]]
            # print(u[j][i].shape)

    for j in range(T - 1):
        for i in range(1, V):
            if np.linalg.norm(u[j][i]) != 0 and np.linalg.norm(u[j + 1][i]) != 0:
                omega[j][i] = findrot(u[j][i], u[j + 1][i], T, V, M)
                angular_v[j][i] = np.cross(omega[j][i], u[j][i], axis=0)
                # print(findrot(u[j][i], u[j + 1][i], T, V, M).shape)
        if j > 0:
            alpha[j - 1] = omega[j] - omega[j - 1]
            angular_a[j - 1] = u[j + 1] - 2 * u[j] + u[j - 1]
    curl = 2 * angular_a[:, :, [0, 2, 1]]
    div = -2 * (angular_v[:, :, [0, 2, 1]] ** 2)
    curl = np.where(curl == 0, 1, curl)
    curl = np.transpose(curl, (2, 0, 1, 3))
    div = np.where(div == 0, 1, div)
    div = np.transpose(div, (2, 0, 1, 3))
    # return curl, div
    return curl, div


def findrot(u, v, T, V, M):
    """find the axis angle parameters to rotate vector u onto vector v"""
    x = u[0]
    y = u[1]
    z = u[2]
    xx = v[0]
    yy = v[1]
    zz = v[2]
    # q1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # q2 = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v, axis=0)
    w_norm = np.linalg.norm(w)
    # q = (x * xx + y * yy + z * zz) / (abs(u) * abs(v))
    q = np.dot(u.T, v)
    qq = [q[0][0], q[1][1]]
    if w_norm < 1e-6:
        A = np.zeros([3, M])
    else:
        ww = w / w_norm  # 3 2
        # qq = np.dot(u.T, v)
        qq = np.arccos(qq)
        A = ww * qq  # normal vector * angle scalar
    return A


# def ftt(data):  # CTVM
#    C,T,V,M = data.shape
#    data = data.reshape(C, T, V*M)
#   x = data[0, :]
#  y = data[1, :]
# z = data[2, :]
# x1, y1, x_ft1, y_fp1 = test_fft(x)
# x2, y2, x_ft2, y_fp2 = test_fft(y)
# x3, y3, x_ft3, y_fp3 = test_fft(z)
# y_fp1 = y_fp1.tensor()
# data_numpy = np.stack((y_fp1,y_fp2,y_fp3))
# data_numpy = data_numpy.reshape(C,T,V,M)
# return data_numpy

# def test_fft(x):
#    sampling_rate = 64  # 采样率
#    t = np.arange(0, 5.3, 5.3 / sampling_rate)

#    xfp = np.fft.fft(x,axis=1).real
#    freqs = np.fft.fftfreq(t.size)  # 表示频率
# xfp = np.abs(xf) / 32  # 代表信号的幅值，即振幅
#    return t, x, freqs, xfp
def get_spectrum(motion_seq):
    """
motion_seq=[CTVM]
    """
    # Apply the 3D Fast Fourier Transform along the temporal axis (T)
    motion_seq_fft = np.fft.fftn(motion_seq, axes=(1,))

    # Shift the zero-frequency component to the center of the spectrum
    # motion_seq_fft = np.fft.fftshift(motion_seq_fft, axes=(0,))

    # Compute the magnitude and phase components of the shifted Fourier spectrum
    amp_spec = np.abs(motion_seq_fft)
    phase_spec = np.angle(motion_seq_fft)

    return amp_spec, phase_spec


def restore_sequence_from_amp(amp_spec):
    """

    """
    # Compute the complex-valued Fourier spectrum from the amplitude and phase components
    motion_seq_fft = amp_spec * np.exp(1j * np.zeros_like(amp_spec))

    # Shift the zero-frequency component back to the corner of the spectrum
    motion_seq_fft = np.fft.ifftshift(motion_seq_fft, axes=(1,))

    # Apply the 3D Inverse Fourier Transform along the temporal axis (T)
    motion_seq = np.fft.ifftn(motion_seq_fft, axes=(1,))

    # Take the real part of the restored sequence to remove any imaginary artifacts
    motion_seq = np.real(motion_seq)

    return motion_seq


def restore_sequence_from_phase(phase_spec):
    """

    """
    # Compute the complex-valued Fourier spectrum from the amplitude and phase components
    motion_seq_fft = np.ones_like(phase_spec) * np.exp(1j * phase_spec)

    # Shift the zero-frequency component back to the corner of the spectrum
    motion_seq_fft = np.fft.ifftshift(motion_seq_fft, axes=(1,))

    # Apply the 3D Inverse Fourier Transform along the temporal axis (T)
    motion_seq = np.fft.ifftn(motion_seq_fft, axes=(1,))
    # motion_seq = motion_seq * (np.e ** (1j * img1_pha))
    # Take the real part of the restored sequence to remove any imaginary artifacts
    motion_seq = np.real(motion_seq)

    return motion_seq


class PoincareBall:
    def __init__(self, c=1.0):
        """
        初始化 PoincareBall 类。
        Args:
            c (float): 曲率参数，默认为 1.0。
        """
        self.c = c  # 曲率参数
        self.min_norm = 1e-15

    def expmap(self, u, p, c):
        sqrt_c = np.sqrt(c)
        u_norm = np.linalg.norm(u, axis=-1, keepdims=True).clip(self.min_norm)
        second_term = (
                np.tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = np.linalg.norm(sub, axis=-1, keepdims=True).clip(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = np.sqrt(c)
        return 2 / sqrt_c / lam * np.arctanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = np.sqrt(c)
        u_norm = np.clip(np.linalg.norm(u, axis=-1, keepdims=True), a_min=self.min_norm, a_max=None)
        gamma_1 = np.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def tan0(self, u, c):
        return u

    def logmap0(self, p, c):
        sqrt_c = np.sqrt(c)
        p_norm = np.linalg.norm(p, axis=-1, keepdims=True).clip(self.min_norm)
        scale = 1. / sqrt_c * np.arctanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = np.sum(x ** 2, axis=dim, keepdims=True)
        y2 = np.sum(y ** 2, axis=dim, keepdims=True)
        xy = np.sum(x * y, axis=dim, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / np.clip(denom, self.min_norm)

def map_to_poincare_ball(S, c=1.0):
    """
    将输入张量映射到 Poincare Ball 模型中并执行对数映射，然后将结果恢复为原始形状。

    Args:
        S (Tensor): 输入张量，形状为 (C, T, V, M)。
        c (float): 曲率参数，默认为 1.0。

    Returns:
        Tensor: 具有与输入张量相同形状的张量，包含映射到切空间的结果。
    """
    poincare_ball = PoincareBall(c)
    C, T, V, M = S.shape
    S_ob = S.reshape(C* T* V* M)
    # 执行 Poincare Ball 模型的投影操作
    S_poincare = poincare_ball.expmap0(S_ob,c)

    # 将 Poincare Ball 模型中的点映射回切空间
    S_tangent = poincare_ball.tan0(S_poincare,c)

    # 最后，将结果恢复为原始形状
    S_tangent = S_tangent.reshape(C, T, V, M)

    return S_tangent
