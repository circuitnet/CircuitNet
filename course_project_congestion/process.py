import argparse
import re
from pathlib import Path

def parse_data(text_content):
    """
    从文本表中解析数据。
    将文本内容解析为一个元组列表 (iterations, nrms, ssim)。
    """
    data_rows = []
    lines = text_content.strip().split('\n')
    for line in lines:
        if line.startswith('|') and re.search(r'\d', line):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 3:
                try:
                    iterations = int(parts[0])
                    nrms = float(parts[1])
                    ssim = float(parts[2])
                    data_rows.append((iterations, nrms, ssim))
                except ValueError:
                    continue
    return data_rows

def generate_latex_table(filename, data_rows):
    """
    为单组数据生成一个LaTeX表格字符串。
    - NRMS最小值标红。
    - SSIM最大值标红。
    """
    if not data_rows:
        return ""

    # 寻找NRMS最小值和SSIM最大值
    nrms_values = [row[1] for row in data_rows]
    ssim_values = [row[2] for row in data_rows]
    min_nrms = min(nrms_values)
    max_ssim = max(ssim_values)

    # 为LaTeX的caption和label清理文件名
    latex_caption_filename = filename.replace('_', r'\_')
    latex_label = f"tab:{Path(filename).stem.replace('.', '_')}"

    # 开始构建LaTeX表格字符串
    latex_lines = [
        r"\begin{table}[h!]",  # h! 强烈建议“此处”
        r"    \centering",
        f"    \\caption{{Results from {latex_caption_filename}}}",
        f"    \\label{{{latex_label}}}",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        \textbf{Iterations} & \textbf{NRMS} & \textbf{SSIM} \\",
        r"        \midrule",
    ]

    # 格式化每一行数据
    for iterations, nrms, ssim in data_rows:
        # 格式化NRMS：最小值标红
        nrms_str = f"{nrms:.4f}"
        if nrms == min_nrms:
            nrms_str = f"\\textcolor{{red}}{{\\textbf{{{nrms_str}}}}}"

        # 格式化SSIM：最大值标红
        ssim_str = f"{ssim:.4f}"
        if ssim == max_ssim:
            ssim_str = f"\\textcolor{{red}}{{\\textbf{{{ssim_str}}}}}"

        latex_lines.append(f"        {iterations:,} & {nrms_str} & {ssim_str} \\\\")

    # 添加表格页脚
    latex_lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    """
    主函数，用于解析命令行参数并驱动整个流程。
    """
    parser = argparse.ArgumentParser(
        description="Convert experiment result .txt files from a folder into a LaTeX document.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder_path",
        type=Path,
        help="Path to the folder containing the .txt result files."
    )
    args = parser.parse_args()
    folder_path = args.folder_path

    if not folder_path.is_dir():
        print(f"Error: Provided path '{folder_path}' is not a valid directory.")
        return

    # LaTeX文档的序言
    preamble = r"""\documentclass[12pt]{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{caption}
\usepackage{siunitx} % 用于正确对齐数字（如果需要）

\begin{document}
"""
    
    # LaTeX文档的结尾
    postamble = r"\end{document}"
    
    print(preamble)

    # 对文件进行排序以保证输出顺序一致
    txt_files = sorted(folder_path.glob('*.txt'))
    
    if not txt_files:
        print(f"% No .txt files found in '{folder_path}'.")

    for i, txt_file_path in enumerate(txt_files):
        filename = txt_file_path.name
        
        # 打印页面标题（文件名）
        latex_section_title = filename.replace('_', r'\_')
        print(f"\\section*{{{latex_section_title}}}")
        
        # 读取文件内容
        try:
            content = txt_file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"% Error reading file {filename}: {e}")
            continue
            
        # 解析数据
        data = parse_data(content)
        
        # 生成并打印表格
        if data:
            latex_table = generate_latex_table(filename, data)
            print(latex_table)
        else:
            print(f"% No data could be parsed from {filename}.")

        # 在每个表格后强制分页（除了最后一个）
        if i < len(txt_files) - 1:
            print(r"\clearpage")
        
        print("\n% " + "="*78 + " % \n") # 为.tex源文件添加分隔符

    print(postamble)

if __name__ == "__main__":
    main()