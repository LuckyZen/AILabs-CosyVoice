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
import sys
import argparse
import logging

from pydantic import BaseModel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
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

class RequestInfo(BaseModel):
    spk_id: str
    audio_type: str
    tts_text: str

def generate_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    # Generate WAV header for the given parameters
    data_size = 0x7FFF0000  # We'll use a large size since we're streaming
    o = bytes("RIFF", 'ascii')                                 # (4byte) Marks file as RIFF
    o += (data_size + 36).to_bytes(4, 'little')               # (4byte) File size in bytes
    o += bytes("WAVE", 'ascii')                               # (4byte) File type
    o += bytes("fmt ", 'ascii')                               # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, 'little')                          # (4byte) Length of above format data
    o += (1).to_bytes(2, 'little')                           # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2, 'little')                    # (2byte) Number of channels
    o += (sample_rate).to_bytes(4, 'little')                 # (4byte) Sample Rate
    o += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')  # (4byte) Bytes per second
    o += (channels * bits_per_sample // 8).to_bytes(2, 'little')               # (2byte) Block alignment
    o += (bits_per_sample).to_bytes(2, 'little')            # (2byte) Bits per sample
    o += bytes("data", 'ascii')                              # (4byte) Data Chunk Marker
    o += (data_size).to_bytes(4, 'little')                  # (4byte) Data size in bytes
    return o

def generate_data(model_output, audio_type='pcm'):
    if audio_type == 'wav':
        yield generate_wav_header()
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.post("/inference_sft")
async def inference_sft(req: RequestInfo):
    model_output = cosyvoice.inference_zero_shot(req.tts_text, None, None, zero_shot_spk_id=req.spk_id, stream=True)
    return StreamingResponse(generate_data(model_output, req.audio_type), media_type='audio/' + req.audio_type)


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Query(), spk_id: str = Query(), audio_type: str = Query('wav')):
    model_output = cosyvoice.inference_zero_shot(tts_text, None, None, zero_shot_spk_id=spk_id, stream=True)
    return StreamingResponse(generate_data(model_output, audio_type), media_type='audio/' + audio_type)


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


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
                        default=9881)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir, True, True, False, True)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
