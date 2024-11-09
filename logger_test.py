import donkeycar as dk
from parts.camera import Camera
from parts.viewer import Viewer
from parts.process import Process
from parts.onnx import Onnx

V = dk.Vehicle()
V.add(Camera(), outputs=['img'])
V.add(Onnx(), inputs=['img'], outputs=['lane', 'drive'])
V.add(Process(), inputs=['lane', 'drive'], outputs=['left', 'right'])
V.add(Viewer(), inputs=['right'])
V.start(rate_hz=2)
