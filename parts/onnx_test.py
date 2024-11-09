import numpy as np
import shutil
import os
import cv2
import onnxruntime as rt
import time

def Run(sess,img):
    print(f"begin proc")
    st = time.time()
    img_rs=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, 0)  # add a batch dimension
    img = img.astype(np.float32)
    img=img / 255.0

    img_out = sess.run(None, {'input.1': img})

    end = time.time()

    x0=img_out[0]
    x1=img_out[1]

# da = driveable area
# ll = lane lines
    da_predict=np.argmax(x0, 1)
    ll_predict=np.argmax(x1, 1)

    DA = da_predict.astype(np.uint8)[0]*255
    LL = ll_predict.astype(np.uint8)[0]*255
    img_rs[DA>100]=[255,0,0]
    img_rs[LL>100]=[0,255,0]
    print(f"Processed image in: {end-st}")
    
    return img_rs


sess = rt.InferenceSession(
        "model.onnx",
        providers=[('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"})])
print('model loaded')

image_list=os.listdir('images')
shutil.rmtree('results')
os.mkdir('results')
for i, imgName in enumerate(image_list):
    img = cv2.imread(os.path.join('images',imgName))
    img=Run(sess,img)
    cv2.imwrite(os.path.join('results',imgName),img)
