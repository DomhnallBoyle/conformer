d_attn = os.environ.get('ATTN_DIM', 64)
model_size = os.environ.get('MODEL_SIZE', 's')
params_d = {
    's': {
        'num_params': 10.3
        'num_encoder_layers': 16,
        'd_encoder': 144,
        'num_attn_heads': 4,
        'conv_kernel_size': 32,
        'num_decoder_layers': 1,
        'd_decoder': 320
    },
    'm': {
        'num_params': 30.7,
        'num_encoder_layers': 16,
        'd_encoder': 256,
        'num_attn_heads': 4,
        'conv_kernel_size': 32,
        'num_decoder_layers': 1,
        'd_decoder': 640
    },
    'l': {
        'num_params': 118.8,
        'num_encoder_layers': 17,
        'd_encoder': 512,
        'num_attn_heads': 8,
        'conv_kernel_size': 32,
        'num_decoder_layers': 1,
        'd_decoder': 640
    }
}
params = params_d.get(model_size, 's')

