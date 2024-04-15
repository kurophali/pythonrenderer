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
        self.color_buffer = torch.full((height, width, 3), 0.0)
        self.screen_coords = torch.full((height, width, 3), 1.0)
        self.texel_size_x = 1 / width
        self.texel_size_y = 1 / height 
        self.screen_vertical_size = 1
        self.screen_unit_size = self.screen_vertical_size / self.height
        # w.i.p. may need to adjust for odd and even sizes
        

        self.rays = torch.full((height, width, 3), 1.0)
        for height_idx in range(len(self.screen_coords)):
            for width_idx in range(len(self.screen_coords[0])):
                self.screen_coords[height_idx, width_idx, 0] = self.texel_size_x * (width_idx + 1)
                self.screen_coords[height_idx, width_idx, 1] = 1 - self.texel_size_y * (height_idx + 1)
        # self.screen_coords = self.screen_coords.clamp(0,1)
        if constants.DEBUG:
            print('================== uv.x ==================')
            print(self.screen_coords.numpy()[:,:,0])
            print('================== uv.y ==================')
            print(self.screen_coords.numpy()[:,:,1])
            print('================== depth or uv.z ==================')
            print(self.screen_coords.numpy()[:,:,2])

        
        # self.vertices = [(0.5, 1, 0.1), (1, 0, 0.1), (0, 0, 0.1)]
            
    def rasterize_screen(self, positions: torch.Tensor, attributes: torch.Tensor, rasterization_id: Optional[int] = -1):
        ''' 
        positions : (3,3) as (vertex_count, dim_count)
        attributes : (3, n) as (vertex_count, attribute_count)
        rasterizing is rather slow if i'm only using matmuls. can't think of a way to reduce overdraw 
        '''
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
        self.color_buffer = can_draw * interpolated_attributes[:,:,1:4] + (1-can_draw) * self.color_buffer
        self.color_buffer = can_draw * interpolated_attributes[:,:,1:4] + (1-can_draw) * self.color_buffer
        self.screen_coords[:,:,2] = (can_draw * depth + (1-can_draw) * screen_depth).reshape((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH))

        # this is even slower
        # can_draw = torch.clamp(torch.sign(depth - screen_depth), 0, 1) * is_inside > 0
        # can_draw = can_draw.reshape(self.height, self.width)
        # self.color_buffer[can_draw] =  interpolated_attributes[can_draw][:,1:4]
        # self.screen_coords[can_draw][:,2] = depth[can_draw][:,0]

        return (is_inside, interpolated_attributes[:,:,0], interpolated_attributes[:,:,1:])         

    def path_trace(self, triangle_batches: torch.Tensor, attribute_batches: torch.Tensor, 
                 camera_position: torch.Tensor, camera_front: torch.Tensor, camera_right: torch.Tensor, camera_up: torch.Tensor,
                 batch_id: Optional[int] = -1, near_plane_distance = 0.1):
        '''
        solution comes from this blog 
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
        assumes the vertical length of the screen is 1
        position_batches: (triangle_count, vertex_count, dim_count)
        attribute_batches: (triangle_count, vertex_count, attribute_count)
        '''
        v0s = triangle_batches[:,0,:]
        v1s = triangle_batches[:,1,:]
        v2s = triangle_batches[:,2,:]
        v01s = (v1s - v0s)
        v02s = (v2s - v0s)
        normals = torch.cross(v01s, v02s) # (triangle_count, 3)
        triangle_count = normals.shape[0]
        # t is stored in (screen_height, screen_width, triangle_count, 1)
        # pending... some constants can be moved to initialzer functions
        ray_offsets = self.screen_coords - 0.5
        # ??? this cartesian dot product should show you something
        self.rays = camera_front + camera_up * ray_offsets[:,:,1][:,:,None].expand((-1,-1,3)) + camera_right * ray_offsets[:,:,0][:,:,None].expand((-1,-1,3)) # (height, width, 3)
        NdotR = self.rays[:,:,None,:]
        NdotR = NdotR.expand(-1, -1, triangle_count, -1)
        NdotR = torch.mul(NdotR, normals).sum(-1) # (height, width, triangle_count)
        NdotO = torch.mul(normals, camera_position).sum(-1) # (triangle_count)
        d = - torch.mul(triangle_batches[:,0,:], normals).sum(-1)
        ts = - (NdotO + d) / NdotR # (height, width, triangle_count) 
        ray_dirs = self.rays[:,:,None,:].expand(-1,-1,2,-1).permute(3,0,1,2) # to (dim, height, width, triangle_count)
        intersection_points = camera_position + (ray_dirs * ts).permute(1,2,3,0)
        inside = self.get_intersection_data(intersection_points, triangle_batches)
        
        # w.i.p. got bug when rendering multiple triangles. testing on screen
        self.color_buffer = inside[:,:,0].unsqueeze(-1).expand((-1,-1,3)) * intersection_points[:,:,0,:]
        # self.color_buffer = inside[:,:,0].unsqueeze(-1).expand((-1,-1,3))
        # self.color_buffer = intersection_points[:,:,0,:]
        # self.color_buffer = self.screen_coords.clamp(0,1)
        # self.color_buffer = ray_offsets
        

        return inside

    def get_intersection_data(self, intersection_positions: torch.Tensor, # (h, w, triangle_count, 3)
                               triangles: torch.Tensor, # (triangle_count, 3, 3)
                               triangle_attributes: torch.Tensor = None # (triangle_count, 3, attribute_count)
                               ):
        # triangle_count = triangles.shape[0]
        # intersection_positions = intersection_positions[:,:,None,:].expand(-1,-1, triangle_count,-1)
        vp0 = triangles[:,0,:] - intersection_positions
        vp1 = triangles[:,1,:] - intersection_positions
        vp2 = triangles[:,2,:] - intersection_positions
        vp01_area_x2 = torch.linalg.vector_norm(torch.cross(vp0, vp1), dim=3) 
        vp02_area_x2 = torch.linalg.vector_norm(torch.cross(vp0, vp2), dim=3) 
        vp12_area_x2 = torch.linalg.vector_norm(torch.cross(vp1, vp2), dim=3) 
        subtriangles_area_x2 = vp01_area_x2 + vp02_area_x2 + vp12_area_x2

        v01 = triangles[:, 1, :] - triangles[:, 0, :]
        v02 = triangles[:, 2, :] - triangles[:, 0, :]
        triangle_area_x2 = torch.linalg.vector_norm(torch.cross(v01, v02), dim=1)
        outside = subtriangles_area_x2 - triangle_area_x2 - 1e-6 # when close to zero some values can be rounded up
        outside = torch.clamp(torch.sign(outside), 0, 1)
        inside = 1 - outside
        
        # w0 = vp12_area_x2 / triangle_area_x2
        # w1 = vp02_area_x2 / triangle_area_x2
        # w2 = vp01_area_x2 / triangle_area_x2

        # frag_attributes = triangle_area_x2
        return inside

    def clear_buffer(self):
        self.color_buffer.fill_(0)
        self.screen_coords[:,:,2] = 0
        # self.color_buffer[:,:,:] = self.screen_coords
        # rasterized = self.rasterize_unoptimized(self.triangle)
        # self.color_buffer[:,:,0] = rasterized
        # return self.color_buffer.numpy()

    def get_buffer(self):
        return self.color_buffer.numpy()
