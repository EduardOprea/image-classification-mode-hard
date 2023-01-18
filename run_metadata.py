class RunMetadata:
    model_input_size_dict = {"resnet50": 232 }
    def __init__(self, model: str, output_size: int, num_epochs: int, batch_size: int,
    freq_ckpt: int, results_dir : str, lr : float , momentum : float, optimizer: str,
    use_label_smoothing: bool, smooth_rate: float) -> None:
        self.model = model
        self.input_size = self.model_input_size_dict[model]
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.freq_ckpt = freq_ckpt
        self.results_dir = results_dir
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_label_smoothing = use_label_smoothing
        self.smooth_rate = smooth_rate