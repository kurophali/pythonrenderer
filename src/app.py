import cv2
import torch
import time
from renderer import Renderer
from scene import Scene
import constants


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_default_device(device)

    renderer: Renderer = Renderer(constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT, device=device, max_triangle_batch_size=2)
    scene: Scene = Scene(device=device)

    prev_frame_time = 0
    new_frame_time = 0

    # print(screen_coords)
    while True:
        renderer.clear_buffer()
        # renderer.rasterize_screen(scene.positions[1], scene.colors[1])
        # renderer.rasterize_screen(scene.positions[0], scene.colors[0], 0)
        # renderer.rasterize_screen(scene.positions[1], scene.colors[1], 1)

        renderer.screen_space_path_trace(triangle_batches=scene.world_triangle, attribute_batches=scene.colors, camera_position=scene.camera_position,
                                 camera_front=scene.camera_direction, camera_right=scene.camera_right, camera_up=scene.camera_up)


        # display color and add fps
        color_buffer = renderer.get_buffer()
        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = str(int(fps))
        color_buffer = cv2.cvtColor(color_buffer, cv2.COLOR_BGR2RGB) 
        cv2.putText(color_buffer, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow('frame', color_buffer)
        # cv2.imshow(fps, color_buffer)
        key = cv2.waitKey(1) & 0xFF 
        movement_amount = 0.1
        if key == ord('q') or constants.DEBUG: 
            break
        elif key == ord('w'):
            # scene.camera_position += scene.camera_direction * movement_amount
            scene.move_camera_down(-movement_amount)
        elif key == ord('s'):
            # scene.camera_position -= scene.camera_direction * movement_amount
            scene.move_camera_down(movement_amount)
        elif key == ord('a'):
            # scene.camera_position -= scene.camera_right * movement_amount
            scene.move_camera_right(-movement_amount)
        elif key == ord('d'):
            # scene.camera_position += scene.camera_right * movement_amount
            scene.move_camera_right(movement_amount)
        elif key == ord('r'): 
            # scene.camera_position += scene.camera_right * movement_amount
            scene.move_camera_forward(movement_amount)
        elif key == ord('f'): 
            # scene.camera_position += scene.camera_right * movement_amount
            scene.move_camera_backward(movement_amount)
    cv2.destroyAllWindows() 
