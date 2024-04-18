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
        # self.camera_position = torch.tensor([0.5,0.5,0], device=self.device)
        # self.camera_direction = torch.tensor([0,0,-1], device=self.device)
        self.camera_right = torch.tensor([1,0,0], device=self.device)
        self.camera_up = torch.tensor([0,1,0], device=self.device)
        self.world_triangle = torch.tensor([[(0, 1, -0.5), (0.5, 0, -0.5), (-0.5, 0, -0.5)],
                                        [(-0.5, 1, 0.5), (0.5, 1, 0.5), (0.0, 0, 0.5)]], device=self.device)
        
        self.camera_position = torch.tensor([0.0,0.0,2.0], device=self.device)
        self.camera_direction = -self.camera_position / self.camera_position.norm()
        # self.world_triangle = torch.tensor([[(0.5, 1, -1), (1, 0, -1), (0, 0, -1)]])
        # self.world_triangle = torch.tensor([[(0.0, 1, -1.5), (1, 1, -1.5), (0.5, 0, -1.5)]])
        self.rotation_matrix = None

    def apply_rotation(self, rotation_matrix):
        self.camera_position = torch.matmul(rotation_matrix, self.camera_position)
        self.camera_direction = - self.camera_position / self.camera_position.norm()
        self.camera_right = torch.cross(self.camera_direction, torch.tensor([0.,1.,0]))
        self.camera_up = torch.cross(self.camera_right, self.camera_direction)

    def move_camera_down(self, by_angle: float):
        by_angle = torch.tensor(by_angle)
        rotation_matrix = torch.tensor([[1, 0, 0],
                                        [0, torch.cos(by_angle), -torch.sin(by_angle)],
                                        [0, torch.sin(by_angle), torch.cos(by_angle)]]).T        
        self.apply_rotation(rotation_matrix=rotation_matrix)

    def move_camera_right(self, by_angle: float):
        by_angle = torch.tensor(by_angle)
        rotation_matrix = torch.tensor([[torch.cos(by_angle), 0, torch.sin(by_angle)],
                                        [0, 1, 0],
                                        [-torch.sin(by_angle), 0, torch.cos(by_angle)]]).T
        self.apply_rotation(rotation_matrix=rotation_matrix)

    def move_camera_backward(self, by_amount: float):
        by_amount = torch.tensor(by_amount)
        self.camera_position -= self.camera_direction
        self.camera_direction = - self.camera_position / self.camera_position.norm()

    def move_camera_forward(self, by_amount: float):
        by_amount = torch.tensor(by_amount)
        self.camera_position += self.camera_direction
        self.camera_direction = - self.camera_position / self.camera_position.norm()
    
    def roll_camera_counterclock(self, by_angle: float):
        rotation_matrix = torch.tensor([[torch.cos(by_angle), torch.sin(by_angle), 0],
                                        [-torch.sin(by_angle), torch.cos(by_angle), 0],
                                        [0,0,1]]).T
        self.apply_rotation(rotation_matrix=rotation_matrix)

