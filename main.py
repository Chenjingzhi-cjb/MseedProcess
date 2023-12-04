import os

import numpy as np
import xlsxwriter
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
    workbook = xlsxwriter.Workbook(folder_path + '\\' + "result.xlsx")
    worksheet = workbook.add_worksheet()

    col = 0
    for filename in os.listdir(folder_path):
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

            row = 0
            for i in range(len(freq_axis)):
                # 往表格写入内容
                worksheet.write(row, col, freq_axis[i])
                worksheet.write(row, col + 1, half_abs_fx[i])
                row += 1
            col += 2

    # 保存
    workbook.close()


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


def main():
    folder_path = f"{os.path.abspath('.')}\data"
    format_conversion(folder_path)
    data_alignment(folder_path, 90000)
    data_conversion(folder_path)
    print("Finished!")


if __name__ == '__main__':
    main()
