from modelscope import snapshot_download

# CosyVoice2-0.5B: 500M参数的语音合成模型，支持高质量语音生成
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# CosyVoice-300M: 300M参数的通用语音合成基础模型
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')

# CosyVoice-300M-25Hz: 300M参数的低采样率(25Hz)语音合成模型，适用于低带宽场景
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')

# CosyVoice-300M-SFT: 300M参数的监督微调版本，语音质量更稳定
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

# CosyVoice-300M-Instruct: 300M参数的指令微调版本，支持语音风格控制
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')

# CosyVoice-ttsfrd: 语音合成前端模型，负责文本处理和特征提取
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')