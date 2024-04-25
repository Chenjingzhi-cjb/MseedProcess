import os

import sys
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read


def format_conversion(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mseed"):
            file_path = os.path.join(folder_path, filename)

            st = read(file_path)
            trace_num = len(st)

            if trace_num == 1:
                st.write(file_path + '_TSPAIR', format='TSPAIR')
                count = count + 1
                print('%04d --------> %s' % (count, filename))
            else:
                print('ERROR:the number of the trace is not one ??? %s' % filename)


def data_conversion(folder_path):
    for index, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".mseed_TSPAIR"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as f:
                raw_datas = f.readlines()
                datas = [int(i.split(' ')[-1]) for i in raw_datas[1:]]

            # 输入信号
            x = np.array(datas)

            # 获取采样率（Fs）的值
            Fs = 1000  # 假设采样率为 1000 Hz
            # 计算频率步长
            freq_step = Fs / len(x)
            # 创建频率轴
            freq_axis = np.arange(0, Fs / 2, freq_step)
            # 零填充输入信号以匹配频率轴的长度
            padded_x = np.pad(x, (0, len(freq_axis) * 2 - len(x)), mode='constant')

            # 计算傅里叶变换
            fx = np.fft.fft(padded_x)

            # 取前半部分复数的绝对值
            half_abs_fx = abs(fx[:len(freq_axis)])

            # 绘制散点图并将图像存储到列表中
            plt.figure(index)
            plt.plot(freq_axis[1:], half_abs_fx[1:], linestyle='-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('FFT Spectrum for {}'.format(filename))  # 使用文件名作为标题
            # plt.plot()

    # 显示所有图像
    plt.show()


def data_alignment(folder_path, number):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mseed_TSPAIR"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            if len(lines) > number:
                lines = lines[:number]

            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)


def main(alignment_number):
    folder_path = f"{os.path.abspath('.')}\data"
    format_conversion(folder_path)
    data_alignment(folder_path, alignment_number)
    data_conversion(folder_path)
    print("Finished!")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(32500)
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Argument error!")
