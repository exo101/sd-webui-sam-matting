"""
SAM Matting 自动安装脚本
Copyright (C) 2024

此脚本用于在 WebUI 启动时自动安装 SAM Matting 所需的依赖包

所需依赖:
- rembg: 背景移除工具
- onnxruntime-gpu: GPU 加速的 ONNX 运行时
- litelama: 轻量级图像修复模型
- segment-anything: Meta 的分割一切模型

使用方法:
将本脚本放置在插件目录的 scripts 文件夹中，WebUI 启动时会自动执行
"""

import os
import sys
import subprocess
from pathlib import Path


def get_python_executable():
    """获取当前 Python 可执行文件路径"""
    python_exe = sys.executable
    if not python_exe:
        # 尝试常见路径
        venv_dir = Path(__file__).parent.parent.parent / "venv"
        if venv_dir.exists():
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = "python"
    return python_exe


def is_package_installed(package_name):
    """检查包是否已安装"""
    try:
        # 处理带额外说明的包名
        base_name = package_name.split('[')[0].split('==')[0].strip()
        
        # 特殊处理：某些包的导入名和包名不一致
        import_map = {
            "onnxruntime-gpu": "onnxruntime",
            "onnxruntime_cpu": "onnxruntime",
            "segment-anything": "segment_anything",
            "litelama": "litelama",
        }
        
        # 先尝试使用映射表中的导入名
        import_name = import_map.get(base_name.lower(), base_name.replace("-", "_").replace(".", "_"))
        
        # 尝试导入
        __import__(import_name)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def install_dependencies():
    """安装所有必需的依赖包"""
    # 定义需要安装的包及其显示名称
    packages = [
        ("rembg", "rembg"),
        ("onnxruntime-gpu", "onnxruntime-gpu"),
        ("litelama", "litelama"),
        ("segment-anything", "segment-anything"),
    ]
    
    installed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for package, display_name in packages:
        # 检查是否已安装
        if is_package_installed(package.split('[')[0]):
            skipped_count += 1
            continue
        
        # 安装包（静默模式）
        python_exe = get_python_executable()
        
        try:
            # 不捕获输出，直接显示到控制台，避免编码问题
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package, "--upgrade"],
                check=True,
                encoding='utf-8',
                errors='ignore'  # 忽略无法解码的字符
            )
            
            if result.returncode == 0:
                installed_count += 1
            else:
                failed_count += 1
                
        except subprocess.CalledProcessError:
            failed_count += 1
        except Exception:
            failed_count += 1
    
    # 只在有安装失败或新安装时显示汇总
    if failed_count > 0 or installed_count > 0:
        print(f"\n{'='*60}")
        print("📊 SAM Matting 依赖安装完成")
        print(f"✅ 新安装：{installed_count} 个")
        print(f"⏭️  已存在：{skipped_count} 个")
        
        if failed_count > 0:
            print(f"❌ 失败：{failed_count} 个")
            print("\n⚠️  以下包安装失败，请手动安装:")
            for package, display_name in packages:
                if not is_package_installed(package.split('[')[0]):
                    print(f"  python -m pip install {package}")
        else:
            print("🎉 所有依赖安装成功！")
        
        print(f"{'='*60}\n")
    
    return failed_count == 0


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SAM Matting 插件 - 自动安装脚本")
    print("="*60)
    print(f"Python 版本：{sys.version}")
    print(f"Python 路径：{sys.executable}")
    print(f"工作目录：{os.getcwd()}")
    print("="*60 + "\n")
    
    # 执行安装
    success = install_dependencies()
    
    if success:
        print("✨ 安装过程顺利完成，即将启动 WebUI...\n")
    else:
        print("⚠️  部分依赖安装失败，但将继续启动 WebUI...\n")
        print("您可以在 WebUI 启动后，打开终端手动安装失败的包\n")
