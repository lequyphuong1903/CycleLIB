class param():
    def __init__(self, device, train_dir, val_dir, batch_size, lr, lambda_id, lambda_cy, num_workers, num_epochs, transforms):
        super().__init__()
        self.device = device
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_id = lambda_id
        self.lambda_cy = lambda_cy
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.transforms = transforms
    