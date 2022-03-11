config = {
        "device": "cuda:0",
    "n_epochs": 150,
    "train_batch_size": 4,
    "logging_step": 1000,
    "padding_length": 250000,
    "max_length": 250000,
    "sample_rate": 16000,
    "lr": 1e-5,
    "momentum": 0.9,
    "max_text_length": 200,
    "queue_size": 65536,
    "text_model": "roberta-base",
    "audio_model": "facebook/hubert-base-ls960",
    "recache": False
}
