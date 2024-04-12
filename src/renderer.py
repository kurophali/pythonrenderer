from typing import Optional
import torch
import constants


class Shader(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class Renderer:        
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.screen_buffer = torch.full((height, width, 3), 0.0)
        self.screen_coords = torch.full((height, width, 3), 1.0)
        self.texel_size_x = 1 / width
        self.texel_size_y = 1 / height 

        for height_idx in range(len(self.screen_coords)):
            for width_idx in range(len(self.screen_coords[0])):
                self.screen_coords[height_idx, width_idx, 0] = self.texel_size_x * (width_idx + 1)
                self.screen_coords[height_idx, width_idx, 1] = 1 - self.texel_size_y * (height_idx + 1)

        if constants.DEBUG:
            print('================== uv.x ==================')
            print(self.screen_coords.numpy()[:,:,0])
            print('================== uv.y ==================')
            print(self.screen_coords.numpy()[:,:,1])
            print('================== depth or uv.z ==================')
            print(self.screen_coords.numpy()[:,:,2])

        
        # self.vertices = [(0.5, 1, 0.1), (1, 0, 0.1), (0, 0, 0.1)]

    def format_vertex_input(self, input_format: dict):
        # re-format the input so we dont need hashing
        self.vertex_input_formatter = []
        for input_name, num_count in input_format:
            self.vertex_input_formatter.append(num_count)
            
    def rasterize(self, positions: torch.Tensor, attributes: torch.Tensor, rasterization_id: Optional[int] = -1):
        # calculate triangle area vs point-corner area
        # if equal then we're inside
        # triangle_area_x2= torch.abs(torch.det(torch.cat((triangle[0:2, 0:2], triangle[1:3, 0:2])).reshape(3,2,2)))
        attribute_size = attributes.shape[-1]
        triangle_area_x2 = torch.abs(torch.det(positions[0:2, 0:2] - positions[1:3, 0:2]))
        attributes = torch.cat((positions[:,2].unsqueeze(-1), attributes), 1) # add depth
        v0s = self.screen_coords[:,:,:2] - positions[0,:2]
        v1s = self.screen_coords[:,:,:2] - positions[1,:2]
        v2s = self.screen_coords[:,:,:2] - positions[2,:2]
        v01_area_x2 = torch.abs(torch.det(torch.cat((v0s, v1s), 2).reshape(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 2, 2)))
        v02_area_x2 = torch.abs(torch.det(torch.cat((v0s, v2s), 2).reshape(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 2, 2)))
        v12_area_x2 = torch.abs(torch.det(torch.cat((v1s, v2s), 2).reshape(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 2, 2)))
        summed_triangle_areas = v01_area_x2 + v02_area_x2 + v12_area_x2
        outside = summed_triangle_areas - triangle_area_x2 - 1e-6 # when close to zero some values can be rounded up
        outside = torch.clamp(torch.sign(outside), 0, 1)

        # unsqueeze(-1) to make (x,y) shaped to (x,y,1)
        areas = torch.cat((v01_area_x2.unsqueeze(-1), v02_area_x2.unsqueeze(-1), v12_area_x2.unsqueeze(-1)), 2)
        areas = torch.permute(areas, (2,0,1)).reshape(3, -1)
        total_area = triangle_area_x2
        interpolated_attributes = torch.matmul(attributes.T, areas).reshape(attribute_size + 1, constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH).permute(1,2,0)
        interpolated_attributes = interpolated_attributes / total_area

        if constants.DEBUG:
            print('================== outside ==================')
            print(outside.numpy())
            # print('================== triangle_area_x2 ==================')
            # print(summed_triangle_areas.numpy())
        
        is_inside = (1 - outside).unsqueeze(-1)
        depth = interpolated_attributes[:,:,0].unsqueeze(-1)
        screen_depth = self.screen_coords[:,:,2].unsqueeze(-1)
        can_draw = torch.clamp(torch.sign(depth - screen_depth), 0, 1) * is_inside # draws closer one a.k.a larger depth so no need to 1 - tensor
        alpha = interpolated_attributes[:,:,4].unsqueeze(-1)
        # didnt know why alpha didnt work. but don't need to blend alpha right now anyway
        # self.screen_buffer = can_draw * alpha * interpolated_attributes[:,:,1:4] + (1-can_draw) * (1-alpha) * self.screen_buffer
        self.screen_buffer = can_draw * interpolated_attributes[:,:,1:4] + (1-can_draw) * self.screen_buffer
        self.screen_coords[:,:,2] = (can_draw * depth + (1-can_draw) * screen_depth).reshape((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH))
        return (is_inside, interpolated_attributes[:,:,0], interpolated_attributes[:,:,1:])         

    def clear_buffer(self):
        self.screen_buffer.fill_(0)
        self.screen_coords[:,:,2] = 0
        # self.screen_buffer[:,:,:] = self.screen_coords
        # rasterized = self.rasterize_unoptimized(self.triangle)
        # self.screen_buffer[:,:,0] = rasterized
        # return self.screen_buffer.numpy()

    def get_buffer(self):
        return self.screen_buffer.numpy()
