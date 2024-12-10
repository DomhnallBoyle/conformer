import os

d_features = 80
max_len = 5000
p_drop = 0.1
model_size = os.environ.get('MODEL_SIZE', 's')
params_d = {
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
