# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torchaudio
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

def text_generator(user_text):
    '''
    # 按多个符号拆分句子
    
    '''
    
    pattern = r'[。！？，；：]'
    text_list = re.split(pattern, user_text)
    for text in text_list:
        if text.strip():
            yield text

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def generate_data_wav(model_output):
    '''
    # 生成wav文件
    '''
    for i, j in enumerate(model_output):
        filename = f"zero_shot_{i}.wav"
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        with open(filename, "rb") as audio_file:
            tts_audio = audio_file.read()
            os.remove(filename)
            yield tts_audio

@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    '''
    按照传入的说话人音频文件，进行音频克隆
    缓存说话人信息，在每次请求时，判断是否有相同的prompt_text，如果有，则直接使用，否则进行音频克隆
    目的增加生成速度
    '''
    # speeck_dict_text内是否包含prompt_text
    my_zero_shot_spk = prompt_text
    if my_zero_shot_spk not in speeck_dict_text:
        print('add zero shot spk:', my_zero_shot_spk)
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, my_zero_shot_spk) is True
        speeck_dict_text.append(my_zero_shot_spk)
        cosyvoice.save_spkinfo()
    # model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    model_output = cosyvoice.inference_zero_shot(text_generator(tts_text), '', '', zero_shot_spk_id=my_zero_shot_spk, stream=False)
    return StreamingResponse(generate_data_wav(model_output))


@app.get("/inference_zero_shot2")
@app.post("/inference_zero_shot2")
async def inference_zero_shot2(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data_wav(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
