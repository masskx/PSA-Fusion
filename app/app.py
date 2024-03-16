import gradio as gr
import cv2
import numpy
import numpy as np

# 模型选择
options = ["ours", "GAN", "DCGAN", "Vgg", "Vgg16", "Vgg19", " Resnet50", "Resnet101", " Resnet152"]
my_example = [
    ["image/vi/1.jpg", "image/ir/1.jpg", "ours"],
    ["image/vi/2.jpg", "image/ir/2.jpg", "ours"],
    ["image/vi/3.jpg", "image/ir/3.jpg", "ours"],
    ["image/vi/4.jpg", "image/ir/4.jpg", "ours"],
    ["image/vi/5.jpg", "image/ir/5.jpg", "ours"],
]


# 推理融合函数
def predict_image(img01, img02, opt):
    combine = cv2.addWeighted(img01, 0.5, img02, 0.5, 0)
    return combine


# 副标题
centered_description = "<div style='text-align: center;'>这是一个图像融合的Demo</div>"
demo = gr.Interface(
    title=' 基于GAN的红外与可见光图像融合方法研究',
    description=centered_description,
    fn=predict_image,
    inputs=[gr.Image(label="可见光"), gr.Image(label="红外光"), gr.Dropdown(choices=options, value=options[0])],
    outputs=gr.Image(label="融合结果"),
    examples=my_example,
    # interpretation='default',
    live=True
)
demo.launch()
