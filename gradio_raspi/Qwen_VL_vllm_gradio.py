import gradio as gr
import requests
import json
import requests
import json
import base64
from PIL import Image
from io import BytesIO
import time
from gradio_image_prompter import ImagePrompter
import gradio as gr
import cv2
import numpy as np

ocr_model_name= "Qwen/Qwen2-VL-7B-Instruct"
ocr_host = "192.168.31.10:8000"

asr_model_name = "Qwen/Qwen2-Audio-7B-Instruct"
asr_host = "192.168.31.10:8000"

chat_model_name = "getfit/DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic"
chat_host = "192.168.31.120:8000"

pipe = "libcamerasrc ! video/x-raw,width=1920,height=1080,framerate=30/1 ! appsink"
cap = cv2.VideoCapture(pipe)
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def video_frame_generator():
    i=0
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    while True:
        i=i+1
        ret, frame = cap.read()
        if not ret:
            break
        # 将 BGR 图像转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将frame上下翻转
        frame = cv2.flip(frame, -1)
        # 调整因为摄像视角带来的畸变
        # 原图中选定的四个点（这些点是你选择的角点）
        pts1 = np.float32([[0, 0], [1920, 0], [300, 1080], [1620, 1080]])

        # 目标图像的四个点，通常是矩形的四个角
        pts2 = np.float32([[0, 0], [1920, 0], [0, 1580], [1920, 1580]])

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # 应用透视变换
        result = cv2.warpPerspective(frame, matrix, (1920, 1580))

        if i%fps==0:
            yield result

    # ret, frame = cap.read()
    # if not ret:
    #     raise RuntimeError("无法打开摄像头")
    # # 将 BGR 图像转换为 RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # 将frame上下翻转
    # frame = cv2.flip(frame, -1)
    # # 调整因为摄像视角带来的畸变
    # # 原图中选定的四个点（这些点是你选择的角点）
    # pts1 = np.float32([[0, 0], [1920, 0], [300, 1080], [1620, 1080]])

    # # 目标图像的四个点，通常是矩形的四个角
    # pts2 = np.float32([[0, 0], [1920, 0], [0, 1580], [1920, 1580]])

    # # 计算透视变换矩阵
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # # 应用透视变换
    # result = cv2.warpPerspective(frame, matrix, (1920, 1580))

    # return result

def image_ORC(image):
    # print(chat_history)
    # print(chat_history_display)
    # print(image)
    image_data = None
    url = "http://"+ocr_host+"/v1/chat/completions"
    
    system = {
                "role": "system",
                "content": [
                    # {"type": "text", "text": "作为一个图像OCR和图像识别服务，请准确识别图片中的文字和表格，图片中的图形内容请用自然语言详细描述图形内容。最终直接返回所有内容。"},
                    {"type": "text", "text": "作为一个习题辅导员，请精准描述图片中的题目内容，对图形的描述尽可能精确和详细。"},
                ],
            }

    if image:
        image_data = encode_image(image)
        user = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpg;base64," + image_data},
                },
            ],
        }
        
    else:
        return

    data = {
    "model": ocr_model_name,
    "messages":[system]+[user],
    "temperature":0,  # 控制生成的随机性
    "max_tokens":2000,   # 控制生成文本的长度
    "stream":True
    }        

    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        content=""
        print("Streaming response:")
        # 按行读取流式响应
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:  # 跳过空行
                if "data: [DONE]" == chunk:
                    print("\nDone!")
                else:
                    delta_data = json.loads(chunk[len("data: "):])
                    if "stop" != delta_data.get("finish_reason"):
                        delta = delta_data.get("choices")[0].get("delta").get("content")
                        if delta=="":
                            continue
                        # print(assistant)
                        content = content + delta
                        yield content
    else:
        print(f"Error: {response.status_code}, {response.text}")

def ASR(audio):
    audio_data = None
    url = "http://"+asr_host+"/v1/chat/completions"
    
    system = {
                "role": "system",
                "content": [
                    {"type": "text", "text": "As an ASR service, generate caption of audio with the audio language."},
                ],
            }

    if audio:
        audio_data = encode_audio(audio)
        user = {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "image_url": {"url": "data:audio/ogg;base64," + audio_data},
                },
            ],
        }
        
    else:
        return

    data = {
    "model": asr_model_name,
    "messages":[system]+[user],
    "temperature":0,  # 控制生成的随机性
    "max_tokens":2000,   # 控制生成文本的长度
    "stream":True
    }        

    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        content=""
        print("Streaming response:")
        # 按行读取流式响应
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:  # 跳过空行
                if "data: [DONE]" == chunk:
                    print("\nDone!")
                else:
                    delta_data = json.loads(chunk[len("data: "):])
                    if "stop" != delta_data.get("finish_reason"):
                        delta = delta_data.get("choices")[0].get("delta").get("content")
                        if delta=="":
                            continue
                        # print(assistant)
                        content = content + delta
                        yield content
    else:
        print(f"Error: {response.status_code}, {response.text}")


def chat_with_image(image, orc, prompt, chat_history):
    # print(chat_history)
    # print(chat_history_display)
    # print(image)
    image_data = None
    url = "http://"+chat_host+"/v1/chat/completions"
    
    system = {
                "role": "system",
                "content": [
                    {"type": "text", "text": "作为一个老师，辅助用户解答各学科的习题，按<思考>string</思考>\n<答案>string</答案>返回思考过程和最后结果。"},
                ],
            }

    if image:
        # image_data = encode_image(image)
        user = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": orc,
                },
                {"type": "text", "text": prompt},
            ],
        }
        
    else:
        user = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }

    if chat_history!=[]:
        data = {
        "model": chat_model_name,
        "messages":[system]+chat_history+[user],
        "temperature":0,  # 控制生成的随机性
        "max_tokens":2000,   # 控制生成文本的长度
        "stream":True
        }
    else:
        data = {
        "model": chat_model_name,
        "messages":[system]+[user],
        "temperature":0,  # 控制生成的随机性
        "max_tokens":2000,   # 控制生成文本的长度
        "stream":True
        }        

    # print(data)

    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        assistant={
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                }

        print("Streaming response:")
        # 按行读取流式响应
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:  # 跳过空行
                if "data: [DONE]" == chunk:
                    print("\nDone!")
                else:
                    delta_data = json.loads(chunk[len("data: "):])
                    if "stop" != delta_data.get("finish_reason"):
                        delta = delta_data.get("choices")[0].get("delta").get("content")
                        if delta=="":
                            continue
                        # print(assistant)
                        content = assistant.get("content")[0].get("text")+delta
                        assistant={
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": content},
                            ],
                        }
                        chat_history=data.get("messages")[1:].copy()
                        chat_history.append(assistant)

                        yield chat_history,convert_messages(chat_history)
    else:
        print(f"Error: {response.status_code}, {response.text}")

def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")  # 根据图片格式选择适当的格式，如 PNG、JPEG
    binary_data = buffer.getvalue()  # 获取二进制数据
    buffer.close()
        # 二进制图片文件转base64编码
    image_data = base64.b64encode(binary_data).decode("utf-8")
    return image_data

def encode_audio(audio):
    buffer = BytesIO()
    audio.save(buffer, format="WAV")  # 根据图片格式选择适当的格式，如 PNG、JPEG
    binary_data = buffer.getvalue()  # 获取二进制数据
    buffer.close()
        # 二进制图片文件转base64编码
    audio_data = base64.b64encode(binary_data).decode("utf-8")
    return audio_data

def decode_image(base64_str):
    # 去除可能存在的前缀
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    
    # 解码 Base64 字符串
    image_data = base64.b64decode(base64_str)
    
    # 将二进制数据转换为 PIL.Image
    image = Image.open(BytesIO(image_data))
    return image

def clear_image():
    return None,None

def clear_history():
    return []

# def predict(im):
#     return im["composite"]

def crop_image(prompts):
    image = prompts["image"]
    points = prompts["points"]
    print(points)
    if image is not None and points is not None and len(points) > 0:
        for point in points:
            x1, y1, x2, y2 = int(point[0]), int(point[1]), int(point[3]), int(point[4])
            cropped_image = image[y1:y2,x1:x2]
        return cropped_image
    return None

def convert_messages(chat_history):
    chatbox=[]
    for chat in chat_history:
        for chat_content in chat.get("content"):
            content_type = chat_content.get("type")
            if content_type == "image_url":
                content = chat_content.get("image_url").get("url")
                image = decode_image(content)

                message = {
                        "role": chat.get("role"),
                        "content": gr.Image(image)
                        }
            else:
                content = chat_content.get("text")
                message = {
                        "role": chat.get("role"),
                        "content": content
                        }
            chatbox.append(message)

    return chatbox

def update_image():
    for frame in video_frame_generator():
        yield {"image": frame, "points": list()}

# 使用 Gradio 的 State 组件保存聊天记录
with gr.Blocks() as demo:
    crop_area = gr.State(None)  # 用于保存框选区域
    chat_history = gr.State(value=[])
    chat_history_display = gr.State(value=[])
    chatbox = gr.Chatbot(label="Chat History",type="messages")
    #增加一个图片的拖拽上传

    # with gr.Row():
    #     im = gr.ImageEditor(
    #         type="pil",
    #         # transforms=("crop", "rotate", "flip", "resize"),
    #         # crop_size="1:1",
    #         show_fullscreen_button=True
    #     )
    
    # im_preview = gr.Image(type="pil",interactive=True)
    im_preview = ImagePrompter(show_label=False,interactive=True)
    
    with gr.Row():
        shot_button = gr.Button("拍照")
        crop_button = gr.Button("截图")

    im = gr.Image(type="pil",label="Chat Image",interactive=False)
    image_ocr = gr.Textbox(label="OCR reuslt:", value="")
    
    input_audio = gr.Audio(
        label="Input Audio",
        sources=["microphone"],
        type="numpy",
        streaming=False,
        waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )

    process_audio

    message = gr.Textbox(label="Your message:", value="")
    # 裁剪按钮
    

    input_audio.stop_recording(ASR, input_audio, message)

    chat_image_button = gr.Button("Send with Image")
    keep_chat_button = gr.Button("Keep Chat")

    shot_button.click(update_image, inputs=None, outputs=im_preview)
    crop_ocr = crop_button.click(crop_image, inputs=im_preview, outputs=im)
    crop_ocr.then(image_ORC,inputs=im, outputs=image_ocr).then(clear_history, outputs=chat_history)

    chat = chat_image_button.click(chat_with_image, inputs=[im, image_ocr, message, chat_history], outputs=[chat_history, chatbox])
    chat.then(clear_image,outputs=[im,message])
    
    chat_submit = message.submit(chat_with_image, inputs=[im, image_ocr, message, chat_history], outputs=[chat_history,chatbox])
    chat_submit.then(clear_image,outputs=[im,message])

    # demo.load(update_image, inputs=None, outputs=im_preview)

# demo.queue()
demo.launch(server_name='0.0.0.0', server_port=40001)
