## 命令参数

1. 数据对齐时长（-A），单位 ms，默认为 40000

2. 执行效果（-E）：
   1. 生成频谱图并立即显示（Plot1），默认
   1. 生成频谱图并单独保存（Plot2）
   2. 导出数据到 Excel 表格（Excel），保存在 "data/result.xlsx"



## 操作流程及命令示例

1. 将 "*.mseed" 文件放入 "data/" 目录下

2. 在当前目录下运行 cmd

   => python main.py （默认数据对齐时长为 40 s，生成频谱图并立即显示）

   => python main.py -A 90000 （设置数据对齐时长为 90 s）

   => python main.py -E Excel （导出数据到 Excel 表格）

