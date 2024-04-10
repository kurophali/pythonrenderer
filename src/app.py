import cv2
import torch
import time

DEBUG = False
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 270

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

        self.custom_init()

        if DEBUG:
            print('================== uv.x ==================')
            print(self.screen_coords.numpy()[:,:,0])
            print('================== uv.y ==================')
            print(self.screen_coords.numpy()[:,:,1])
            print('================== depth or uv.z ==================')
            print(self.screen_coords.numpy()[:,:,2])

    def custom_init(self):
        self.triangle = torch.Tensor([(0.5, 1, 0.1), (1, 0, 0.1), (0, 0, 0.1)])

        # self.vertices = [(0.5, 1, 0.1), (1, 0, 0.1), (0, 0, 0.1)]


    def sign(self, p0s, p1, p2):
        return (p0s[:,:,0] - p2[0]) * (p0s[:,:,1] - p2[1]) - (p1[0] - p2[0]) * (p0s[:,:,1] - p2[1])

    def rasterize(self, triangle: torch.Tensor):
        d0 = self.sign(self.screen_coords, triangle[0], triangle[1])
        d1 = self.sign(self.screen_coords, triangle[1], triangle[2])
        d2 = self.sign(self.screen_coords, triangle[2], triangle[0])
        # has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
        # has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)
        has_neg = (1 - torch.sign(d0)) + (1 - torch.sign(d1)) + (1 - torch.sign(d2))
        has_neg = torch.clamp(torch.sign(has_neg), 0, 1)
        has_pos = torch.sign(d0) + torch.sign(d1) + torch.sign(d2)
        has_pos = torch.clamp(torch.sign(has_pos), 0, 1)

        is_in_triangle = 1 - (has_neg * has_pos)

        if DEBUG:
            print('================== has_neg ==================')
            print(has_neg.numpy())
            print('================== has_pos ==================')
            print(has_pos.numpy())
            print('================== inside ==================')
            print(is_in_triangle.numpy())
                
        return is_in_triangle
    
    def rasterize_unoptimized(self, triangle: torch.Tensor):
        # calculate triangle area vs point-corner area
        # if equal then we're inside
        # triangle_area_x2= torch.abs(torch.det(torch.cat((triangle[0:2, 0:2], triangle[1:3, 0:2])).reshape(3,2,2)))
        triangle_area_x2 = torch.abs(torch.det(triangle[0:2, 0:2] - triangle[1:3, 0:2]))
        v0s = self.screen_coords[:,:,:2] - self.triangle[0,:2]
        v1s = self.screen_coords[:,:,:2] - self.triangle[1,:2]
        v2s = self.screen_coords[:,:,:2] - self.triangle[2,:2]
        v01_area_x2 = torch.abs(torch.det(torch.cat((v0s, v1s), 2).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 2, 2)))
        v02_area_x2 = torch.abs(torch.det(torch.cat((v0s, v2s), 2).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 2, 2)))
        v12_area_x2 = torch.abs(torch.det(torch.cat((v1s, v2s), 2).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 2, 2)))
        summed_triangle_areas = v01_area_x2 + v02_area_x2 + v12_area_x2
        outside = summed_triangle_areas - triangle_area_x2 - 1e-6
        outside = torch.clamp(torch.sign(outside), 0, 1)
        # W.I.P......
        if DEBUG:
            print('================== outside ==================')
            print(outside.numpy())
            # print('================== triangle_area_x2 ==================')
            # print(summed_triangle_areas.numpy())
        return 1 - outside


    def draw(self):
        self.screen_buffer.fill_(0)

        # self.screen_buffer[:,:,:] = self.screen_coords
        rasterized = self.rasterize_unoptimized(self.triangle)
        self.screen_buffer[:,:,0] = rasterized
        return self.screen_buffer.numpy()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT)

    prev_frame_time = 0
    new_frame_time = 0

    # print(screen_coords)
    while True:
        color_buffer = renderer.draw()

        # display fps
        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = str(int(fps))
        color_buffer = cv2.cvtColor(color_buffer, cv2.COLOR_BGR2RGB) 
        cv2.putText(color_buffer, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow('frame', color_buffer)
        # cv2.imshow(fps, color_buffer)
        if cv2.waitKey(1) & 0xFF == ord('q') or DEBUG: 
            break
    cv2.destroyAllWindows() 
