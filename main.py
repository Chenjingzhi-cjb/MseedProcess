import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from obspy.core import read

import matplotlib
matplotlib.use('TkAgg')


def data_conversion_base(filename, folder_path, alignment_count):
    file_path = os.path.join(folder_path, filename)

    stream = read(file_path)
    trace = stream[0]

    # 输入信号
    x = np.array(trace.data)
    if len(x) > alignment_count:
        x = x[:alignment_count]

    # 获取采样率（Fs）的值
    fs = trace.stats.sampling_rate

    # 创建频率轴
    freq_axis = np.fft.rfftfreq(len(x), d=1/fs)

    # 计算傅里叶变换
    fx = np.fft.rfft(x)

    # 计算 80-180 Hz 的 速度 RMS 值
    velocity = 2 * np.pi * freq_axis * np.abs(fx)  # 计算速度幅值
    freq_mask = (freq_axis >= 8) & (freq_axis <= 80)  # 提取 8 Hz 至 80 Hz 范围
    velocity_in_band = velocity[freq_mask]
    velocity_rms_ums = np.sqrt(np.mean(velocity_in_band ** 2)) / 1000000.0  # 计算 RMS 值

    # 取前半部分复数的绝对值
    half_abs_fx = abs(fx)

    return freq_axis, half_abs_fx


def data_conversion_C(folder_path, alignment_count):
    for index, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".mseed"):
            freq_axis, half_abs_fx = data_conversion_base(filename, folder_path, alignment_count)

            # 绘制散点图并将图像存储到列表中
            plt.figure(index)
            plt.plot(freq_axis[1:], half_abs_fx[1:], linestyle='-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('FFT Spectrum for {}'.format(filename))  # 使用文件名作为标题

    # 显示所有图像
    plt.show()


def data_conversion_S(folder_path, alignment_count):
    workbook = xlsxwriter.Workbook(folder_path + '\\' + "result.xlsx")
    worksheet = workbook.add_worksheet()

    col = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mseed"):
            freq_axis, half_abs_fx = data_conversion_base(filename, folder_path, alignment_count)

            worksheet.write(0, col, filename.removesuffix(".mseed"))

            # 去直流
            row = 1
            for i in range(1, len(freq_axis)):
                # 往表格写入内容
                worksheet.write(row, col, freq_axis[i])
                worksheet.write(row, col + 1, half_abs_fx[i])
                row += 1
            col += 2

    # 保存
    workbook.close()


def main(alignment_count, exec_type):
    folder_path = f"{os.path.abspath('.')}\data"

    if exec_type == 'C':
        data_conversion_C(folder_path, alignment_count)
    elif exec_type == 'S':
        data_conversion_S(folder_path, alignment_count)

    print("Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse arguments for main.py')
    parser.add_argument('-A', type=int, help='Alignment Count')
    parser.add_argument('-E', choices=['C', 'S'], help='Exec Type')

    args = parser.parse_args()

    if args.A is None:
        args.A = 32500

    if args.E is None:
        args.E = 'C'

    main(args.A, args.E)
