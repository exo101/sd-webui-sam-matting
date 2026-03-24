import gradio as gr
from modules import script_callbacks
from pathlib import Path

# 导入各个功能模块
try:
    from .scripts.segment_anything_ui import create_sam_ui, SAM_AVAILABLE
except ImportError:
    try:
        from scripts.segment_anything_ui import create_sam_ui, SAM_AVAILABLE
    except ImportError:
        create_sam_ui = None
        SAM_AVAILABLE = False
        print("Warning: Could not import segment_anything_ui")

try:
    from .scripts.image_matting import create_image_matting_module
except ImportError:
    try:
        from scripts.image_matting import create_image_matting_module
    except ImportError:
        create_image_matting_module = None
        print("Warning: Could not import image_matting")

try:
    from .scripts.cleaner_ui import create_cleaner_module, CLEANER_AVAILABLE
except ImportError:
    try:
        from scripts.cleaner_ui import create_cleaner_module, CLEANER_AVAILABLE
    except ImportError:
        create_cleaner_module = None
        CLEANER_AVAILABLE = False
        print("Warning: Could not import cleaner_ui")


def segmentation_tab():
    """创建图像分割/抠图/清理标签页"""
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            # 智能抠图标签页
            with gr.TabItem("智能抠图"):
                if create_image_matting_module is not None:
                    try:
                        create_image_matting_module()
                    except Exception as e:
                        gr.Markdown(f"智能抠图模块加载失败：{e}")
                else:
                    gr.Markdown("智能抠图模块当前不可用。")
            
            # 图像分割标签页
            with gr.TabItem("图像分割"):
                if SAM_AVAILABLE and create_sam_ui is not None:
                    try:
                        sam_ui_components = create_sam_ui()
                    except Exception as e:
                        with gr.Group():
                            gr.Markdown("## 图像分割")
                            gr.Markdown(f"图像分割模块加载时出现错误：{str(e)}")
                            gr.Markdown("请检查控制台输出以获取详细错误信息。")
                        import traceback
                        traceback.print_exc()
                else:
                    gr.Markdown("图像分割模块不可用。请确保已安装 segment-anything 库。")
            
            # 图像清理标签页
            with gr.TabItem("图像清理"):
                if CLEANER_AVAILABLE and create_cleaner_module is not None:
                    try:
                        cleaner_ui_components = create_cleaner_module()
                    except Exception as e:
                        with gr.Group():
                            gr.Markdown("## 图像清理")
                            gr.Markdown(f"图像清理模块加载时出现错误：{str(e)}")
                            gr.Markdown("请检查控制台输出以获取详细错误信息。")
                        import traceback
                        traceback.print_exc()
                else:
                    gr.Markdown("图像清理模块不可用。请确保已安装 litelama 库。")
    
    return [(ui, "图像分割与清理", "Segmentation_Tab")]


def on_app_started(*args, **kwargs):
    """在 WebUI 启动时显示插件信息"""
    print("=" * 60)
    print("SAM 智能分割与抠图插件 - SAM Matting")
    print()
    print("集成功能：")
    print("- Segment Anything 图像分割")
    print("- Rembg 智能抠图")
    print("- LiteLama 图像清理")
    print()
    print("使用须知：请确保已安装所有必要的依赖库。")
    print("=" * 60)


# 注册标签页和启动事件
script_callbacks.on_ui_tabs(segmentation_tab)
script_callbacks.on_app_started(on_app_started)
