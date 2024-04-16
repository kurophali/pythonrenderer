import torch

class Scene:
    def __init__(self, device) -> None:
        self.device = device
        self.positions = torch.tensor([[(0.5, 1, 0.2), (1, 0, 0.2), (0, 0, 0.2)],
                                        [(0.0, 1, 0.1), (1, 1, 0.1), (0.5, 0, 0.1)]], device=self.device)
        # self.colors = torch.tensor([[(1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 0.2, 1.0, 1.0)],
        #                             [(0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0)]], device=self.device)
        self.colors = torch.tensor([[(1.0, 0.2, 0.2, 1.0), (1.0, 0.2, 0.2, 1.0), (1.0, 0.2, 0.2, 1.0)],
                                    [(0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0)]], device=self.device)
        self.camera_position = torch.tensor([0.5,0.5,0], device=self.device)
        self.camera_direction = torch.tensor([0,0,-1], device=self.device)
        self.camera_right = torch.tensor([1,0,0], device=self.device)
        self.camera_up = torch.tensor([0,1,0], device=self.device)
        self.world_triangle = torch.tensor([[(0.5, 1, -1), (1, 0, -1), (0, 0, -1)],
                                        [(0.0, 1, -1.5), (1, 1, -1.5), (0.5, 0, -1.5)]], device=self.device)
        # self.world_triangle = torch.tensor([[(0.5, 1, -1), (1, 0, -1), (0, 0, -1)]])
        # self.world_triangle = torch.tensor([[(0.0, 1, -1.5), (1, 1, -1.5), (0.5, 0, -1.5)]])
