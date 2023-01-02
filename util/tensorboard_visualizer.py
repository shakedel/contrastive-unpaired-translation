import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class TensorboardVisualizer:

    def __init__(self, logs_dir: str):

        self.writer = SummaryWriter(logs_dir)

    def log_losses(self, losses: dict, step: int):
        for k, v in losses.items():
            self.writer.add_scalar(f"losses/{k}", v, global_step=step)

    def log_images(self, images: dict, step: int):
        images_to_log = []
        for k, v in images.items():
            v = (v.squeeze() + 1) / 2.0
            images_to_log.append(v)

        grid = make_grid(images_to_log)
        self.writer.add_image('sample_images', grid, step)
