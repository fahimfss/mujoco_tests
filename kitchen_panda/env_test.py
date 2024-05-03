import numpy as np
import mujoco 
import cv2


def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid.mp4', fourcc, 25, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main():
    ## Setup
    images = []
    height = 480
    width = 640

    model_path = 'env.xml' 
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    ## Move to grabbing position
    kf = model.keyframe('home')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera='cam')
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    ## Move to grabbing position
    ctrl = kf.qpos[:9] * 0.95  ## The arm keeps moving if not multiplied by 0.95
    ctrl[7] = 0
    ctrl[8] = 0
    for i in range(100):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera='cam')
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    ## Lift up 
    ctrl = kf.qpos[:9] * 0.95
    ctrl[3] = 0
    ctrl[7] = 0
    ctrl[8] = 0
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data, nstep=2)
        renderer.update_scene(data, camera='cam')
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    create_vid(images)

if __name__ == '__main__':
    main()