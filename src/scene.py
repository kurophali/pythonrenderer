import torch

class Scene:
    def __init__(self) -> None:
        self.positions = torch.Tensor([[(0.5, 1, 0.1), (1, 0, 0.1), (0, 0, 0.0)],
                                        [(0.0, 1, 0.1), (1, 1, 0.1), (0.5, 0, 0.0)]])
        self.colors = torch.Tensor([[(1.0, 0.2, 0.2), (1.0, 0.2, 0.2), (1.0, 0.2, 0.2)],
                                    [(0.2, 1.0, 0.2), (0.2, 1.0, 0.2), (0.2, 1.0, 0.2)]])
        
        self.triangles = torch.cat((self.positions, self.colors), 1)
