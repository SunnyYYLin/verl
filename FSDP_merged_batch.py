#!/usr/bin/env python3
"""
批量合并 FSDP 格式的 checkpoint 为 Hugging Face 格式
直接使用已有的 legacy_model_merger 模块

使用方法：
    python batch_merge.py --base_dir /path/to/checkpoints [--output_base /path/to/output] [--model_type jamba] [--suffix _hf]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from legacy_model_merger import BaseModelMerger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_checkpoint(input_dir, output_dir, model_type=None):
    """合并单个 checkpoint"""
    if os.path.exists(output_dir):
        logging.warning(f"输出目录 {output_dir} 已存在，跳过合并（如需覆盖请先手动删除）")
        return

    # 如果未指定 model_type，尝试从 input_dir/config.json 中读取
    if model_type is None:
        config_path = os.path.join(input_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_type = config.get("model_type", "jamba")
            logging.info(f"从 config.json 读取 model_type = {model_type}")
        else:
            model_type = "jamba"
            logging.warning(f"未找到 config.json，使用默认 model_type = {model_type}")

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"开始合并: {input_dir} -> {output_dir}")
    merger = BaseModelMerger(
        model_type=model_type,
        input_dir=input_dir,
        output_dir=output_dir
    )
    merger.merge()
    logging.info(f"合并完成: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="批量合并 FSDP checkpoints")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="包含多个 global_step_* 子目录的根目录（输入目录）")
    parser.add_argument("--output_base", type=str, default=None,
                        help="输出根目录，若不指定则每个 checkpoint 合并到原目录同级加 suffix")
    parser.add_argument("--model_type", type=str, default=None,
                        help="模型类型，若不指定则尝试从 config.json 读取")
    parser.add_argument("--suffix", type=str, default="_hf",
                        help="输出目录后缀，默认为 _hf（仅当 output_base 未指定时生效，若指定 output_base，子目录名保留原名或加上后缀）")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        logging.error(f"目录不存在: {base_dir}")
        sys.exit(1)

    # 查找所有 global_step_* 子目录
    checkpoint_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")]
    if not checkpoint_dirs:
        logging.error(f"在 {base_dir} 中未找到 global_step_* 目录")
        sys.exit(1)

    logging.info(f"找到 {len(checkpoint_dirs)} 个 checkpoint 目录")

    # 如果指定了 output_base，则创建该目录
    if args.output_base:
        output_base = Path(args.output_base).resolve()
        os.makedirs(output_base, exist_ok=True)
    else:
        output_base = None

    for ckpt_dir in sorted(checkpoint_dirs):  # 按名称排序
        input_path = str(ckpt_dir)

        # 生成输出路径
        if output_base is not None:
            # 输出到 output_base 下，子目录名 = 原目录名 + suffix（如果 suffix 不为空）
            output_subdir = ckpt_dir.name + args.suffix
            output_path = str(output_base / output_subdir)
        else:
            # 原逻辑：与原目录同级，加上 suffix
            output_path = str(ckpt_dir) + args.suffix

        try:
            merge_checkpoint(input_path, output_path, args.model_type)
        except Exception as e:
            logging.error(f"合并 {input_path} 时出错: {e}")
            continue

    logging.info("批量合并完成")

if __name__ == "__main__":
    main()