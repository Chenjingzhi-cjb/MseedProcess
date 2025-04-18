import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from obspy.core import read

import matplotlib
matplotlib.use('TkAgg')


def data_conversion_base(filename, folder_path, alignment_time):
    file_path = os.path.join(folder_path, filename)

    stream = read(file_path)
    trace = stream[0]

    # 获取采样率（Fs）的值
    fs = trace.stats.sampling_rate

    # 时间 -> 数量
    alignment_count = int(alignment_time * fs / 1000.0)

    # 输入信号
    x = np.array(trace.data)
    if len(x) > alignment_count:
        x = x[:alignment_count]

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


def trim_freq_data(freq_axis, half_abs_fx, freq_min, freq_max):
    """裁剪频率轴和对应幅值，返回新列表"""
    trimmed = [(f, a) for f, a in zip(freq_axis, half_abs_fx) if freq_min <= f <= freq_max]
    return list(zip(*trimmed)) if trimmed else ([], [])


def data_conv_pyplot(folder_path, alignment_time, cmd, freq_min, freq_max):
    for index, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".mseed"):
            freq_axis, half_abs_fx = data_conversion_base(filename, folder_path, alignment_time)

            if freq_axis[-1] < freq_max:
                freq_max = freq_axis[-1]

            # 裁剪频率范围
            if freq_min != freq_axis[0] or freq_max != freq_axis[-1]:
                freq_axis, half_abs_fx = trim_freq_data(freq_axis, half_abs_fx, freq_min, freq_max)

            if len(freq_axis) == 0:
                continue  # 如果裁剪后为空，跳过该文件

            start_idx = 1 if freq_axis[0] == 0 else 0  # 去直流

            # 绘制散点图并将图像存储到列表中
            plt.figure(index)
            plt.plot(freq_axis[start_idx:], half_abs_fx[start_idx:], linestyle='-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('FFT Spectrum for {}'.format(filename))  # 使用文件名作为标题

            if cmd == "save":
                # 保存图像
                plt.savefig(f"{folder_path}/{filename.removesuffix('.mseed')}.png")

    if cmd == "show":
        # 显示所有图像
        plt.show()


def num_to_excel_col(n):
    """将 0-based 列索引转换为 Excel 列名（A, B, C, ..., Z, AA, AB, ...）"""
    col_name = ""
    while n >= 0:
        col_name = chr(n % 26 + ord('A')) + col_name
        n = n // 26 - 1
    return col_name


def data_conv_excel(folder_path, alignment_time, freq_min, freq_max):
    workbook = xlsxwriter.Workbook(folder_path + '\\' + "result.xlsx")
    worksheet = workbook.add_worksheet()

    chart = workbook.add_chart({"type": "scatter", 'subtype': 'smooth'})

    col = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mseed"):
            # 计算频域数据
            freq_axis, half_abs_fx = data_conversion_base(filename, folder_path, alignment_time)

            # 裁剪频率范围
            if freq_min != freq_axis[0] or freq_max != freq_axis[-1]:
                freq_axis, half_abs_fx = trim_freq_data(freq_axis, half_abs_fx, freq_min, freq_max)

            if len(freq_axis) == 0:
                continue  # 如果裁剪后为空，跳过该文件

            # 写入频域数据
            worksheet.write(0, col, filename.removesuffix(".mseed"))

            start_idx = 1 if freq_axis[0] == 0 else 0  # 去直流
            row = 1
            for i in range(start_idx, len(freq_axis)):
                worksheet.write(row, col, freq_axis[i])
                worksheet.write(row, col + 1, half_abs_fx[i])
                row += 1

            # 添加数据到图表
            x_col = num_to_excel_col(col)
            y_col = num_to_excel_col(col + 1)

            chart.add_series({
                "name": f"={worksheet.name}!${x_col}$1",  # 组名称
                "categories": f"={worksheet.name}!${x_col}$2:${x_col}${row}",  # X 轴数据
                "values": f"={worksheet.name}!${y_col}$2:${y_col}${row}",  # Y 轴数据
                "line": {"width": 1.5, "smooth": True},  # 平滑曲线
            })

            col += 2

    # 设置图表属性
    chart.set_title({"name": "Spectral Analysis"})
    chart.set_x_axis({
        "name": "Frequency (Hz)",
        "min": 0,
        "major_gridlines": {"visible": True, "line": {"width": 0.5, "dash_type": "dash"}},  # 启用主网格线
    })
    chart.set_y_axis({
        "name": "Amplitude",
        "min": 0,
        "major_gridlines": {"visible": True, "line": {"width": 0.5, "dash_type": "dash"}},  # 启用主网格线
    })
    chart.set_size({"width": 1000, "height": 520})

    # 插入图表
    worksheet.insert_chart("B2", chart)

    # 保存
    workbook.close()


def main(alignment_time, exec_type, freq_min, freq_max):
    folder_path = f"{os.path.abspath('.')}\data"

    if exec_type == 'Plot1':
        data_conv_pyplot(folder_path, alignment_time, "show", freq_min, freq_max)
    elif exec_type == 'Plot2':
        data_conv_pyplot(folder_path, alignment_time, "save", freq_min, freq_max)
    elif exec_type == 'Excel':
        data_conv_excel(folder_path, alignment_time, freq_min, freq_max)

    print("Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse arguments for main.py')
    parser.add_argument('-A', type=int, help='Alignment Time')
    parser.add_argument('-E', choices=['Plot1', 'Plot2', 'Excel'], help='Exec Type')
    parser.add_argument('-FMin', type=float, help='Frequency Min')
    parser.add_argument('-FMax', type=float, help='Frequency Max')

    args = parser.parse_args()

    if args.A is None:
        args.A = 40000  # ms

    if args.E is None:
        args.E = 'Excel'

    if args.FMin is None:
        args.FMin = 10.0

    if args.FMax is None:
        args.FMax = 100.0

    main(args.A, args.E, args.FMin, args.FMax)
