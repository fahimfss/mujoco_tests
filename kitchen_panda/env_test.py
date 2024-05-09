import numpy as np
import mujoco 
import cv2
from mujoco.glfw import glfw
import mujoco as mj


HEIGHT = 900
WIDTH = 1200
VID_FPS = 100 
MODEL_PATH = 'env.xml' 
VID_PATH = 'vid.mp4'


def init_cv_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(VID_PATH, fourcc, VID_FPS, (WIDTH, HEIGHT))
    
def create_vid(context, video):
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    viewport = mj.MjrRect(0, 0, WIDTH, HEIGHT)
    mujoco.mjr_readPixels(image, None, viewport, context)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.rotate(image, cv2.ROTATE_180)
    image = cv2.flip(image, 1) 
    video.write(image)

def init_mj_render(model):
    glfw.init()
    window = glfw.create_window(WIDTH, HEIGHT, "test", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    cam = mj.MjvCamera()    
    mj.mjv_defaultCamera(cam)             
    cam.azimuth = 135
    cam.elevation = -5
    cam.distance = 2.0
    cam.lookat = np.array([0.5, 0.0, 0.5])       
    opt = mj.MjvOption()    
    mj.mjv_defaultOption(opt)
    return (context, opt, cam, scene, window)       

def viewport_render(model, data, render_tuple):
    context, opt, cam, scene, window = render_tuple
    viewport = mj.MjrRect(0, 0, WIDTH, HEIGHT)
    mj.mjv_updateScene(model, data, opt, None, cam, 
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

def main():
    ## Setup
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rt = init_mj_render(model)
    vid = init_cv_video()

    ## Move to grabbing position
    kf = model.keyframe('home')
    data.qpos = kf.qpos
    data.ctrl = kf.ctrl 
    mujoco.mj_forward(model, data)
    viewport_render(model, data, rt)
    create_vid(rt[0], vid) 

    ## Grab the cylinder
    ctrl = kf.ctrl
    finger_pos_1 = ctrl[7] 
    finger_pos_2 = ctrl[8] 
    for i in range(255):
        data.ctrl = ctrl
        mujoco.mj_step(model, data) 
        viewport_render(model, data, rt)
        create_vid(rt[0], vid)
        finger_pos_1 -= 0.00015686274
        finger_pos_2 -= 0.00015686274
        ctrl[7] = max(finger_pos_1, 0) 
        ctrl[8] = max(finger_pos_2, 0) 

    ## Lift up 
    joint_4_pos = ctrl[3] 
    for i in range(500):
        data.ctrl = ctrl
        mujoco.mj_step(model, data, nstep=1) 
        viewport_render(model, data, rt) 
        create_vid(rt[0], vid)
        joint_4_pos += 0.003
        ctrl[3] = joint_4_pos 

    vid.release()
    glfw.terminate()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()