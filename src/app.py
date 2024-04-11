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
    if torch.cuda.is_available():
        torch.set_default_device(device)

    renderer: Renderer = Renderer(constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
    scene: Scene = Scene()

    prev_frame_time = 0
    new_frame_time = 0

    # print(screen_coords)
    while True:
        renderer.clear_buffer()
        is_inside, attributes = renderer.rasterize(scene.positions[0], scene.colors[0])

        # act as fragment shader
        renderer.screen_buffer = is_inside * attributes

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
        if cv2.waitKey(1) & 0xFF == ord('q') or constants.DEBUG: 
            break
    cv2.destroyAllWindows() 
