import os

import torch


d_features = 80
max_len = 5_000
p_drop = 0.1
model_size = os.environ.get('MODEL_SIZE', 's')
params_d = {
    'tiny': {
        'num_params': None,
        'num_encoder_layers': 4,
        'd_encoder': 144,
        'num_attn_heads': 4,
        'd_attn': 36,
        'conv_kernel_size': 31,
        'num_decoder_layers': 1,
        'd_decoder': 160
    },
    's': {
        'num_params': 10.3,
        'num_encoder_layers': 16,
        'd_encoder': 144,
        'num_attn_heads': 4,
        'd_attn': 36,
        'conv_kernel_size': 31,
        'num_decoder_layers': 1,
        'd_decoder': 320
    },
    'm': {
        'num_params': 30.7,
        'num_encoder_layers': 16,
        'd_encoder': 256,
        'num_attn_heads': 4,
        'd_attn': 64,
        'conv_kernel_size': 31,
        'num_decoder_layers': 1,
        'd_decoder': 640
    },
    'l': {
        'num_params': 118.8,
        'num_encoder_layers': 17,
        'd_encoder': 512,
        'num_attn_heads': 8,
        'd_attn': 64,
        'conv_kernel_size': 31,
        'num_decoder_layers': 1,
        'd_decoder': 640
    }
}
params = params_d.get(model_size, 's')

# mel filterbanks
window_size = 400  # 25ms window
stride = 160  # 10ms hop
num_mels = d_features

# spec-aug
num_time_masks = 10
frequency_mask = 27
max_time_mask_ratio = 0.05  # mask max size = this * utterance length
min_warp_rate, max_warp_rate = 0.8, 1.2

# training
lr_initial = 1e-3
lr_betas = (0.9, 0.98)
lr_eps = 1e-9
warmup_steps = 10_000
l2_regularisation_weight = 1e-6
pad_value = 0.0
blank = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
