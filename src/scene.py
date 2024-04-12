import torch

class Scene:
    def __init__(self) -> None:
        self.positions = torch.Tensor([[(0.5, 1, 0.2), (1, 0, 0.2), (0, 0, 0.2)],
                                        [(0.0, 1, 0.1), (1, 1, 0.1), (0.5, 0, 0.1)]])
        self.colors = torch.Tensor([[(1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 0.2, 1.0, 1.0)],
                                    [(0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0)]])
        self.camera_position = torch.Tensor([0,0,0])
        self.camera_direction = torch.Tensor([0,0,-1])
        self.camera_right = torch.Tensor([1,0,0])
        self.camera_up = torch.Tensor([0,1,0])
        self.world_triangle = torch.Tensor([[(0.5, 1, -2.5), (1, 0, -2.5), (0, 0, -2.5)],
                                        [(0.0, 1, -1.5), (1, 1, -1.5), (0.5, 0, -1.5)]])