# Real Time Social Distance Detector

Our goal is to build a social distance detector which is fast and accurate enough to run on-device inference on real-time video-feed.

# Introduction

[Check out the Colab Notebook](https://colab.research.google.com/drive/1FbXD9kMwmTE3UW56H41QlWMiNQtZ6Irp?usp=sharing)

In any pandemic where the disease spreads through physical contact, social distancing has always been the most efficient method to counteract the disease. So, in recent times where Covid-19 has devastated the lives of many people, we need to maintain social distance among people. A majority of industries are demanding a tool to determine the extent to which people are following this safety protocol.

To serve this purpose, there are three main points :

1. Real Time Object(Human) Detection : Object Detection is a combination of object classification and object localization. It is trained to detect the presence and location of multiple classes of objects. Here, the object which we are addressing is humans. We need to identify a particular class, that is humans, in real-time video feed. There are various approaches for object detection, it can be `Region Classification Method` with R-CNN or Fast R-CNN, `Implicit Anchor Method` with Faster R-CNN, YOLO v1-v4, or EfficientDet, `Keypoint Estimation Method` with CenteNet or CornerNet.

Our motto is to detect pedestrians from a scene. For on-device inference, we will use `SSD(Single Shot Detector)` model. This model will be converted to TensorFlow Lite from [TensorFlow Lite Object Detection Api(TFOD)](https://github.com/tensorflow/models/tree/master/research/object_detection). The mobile model used here is `SSD_MobileDet_cpu_coco`.

2. Calibration and Transformation
3. Determining social distance violation

The following will be discussed : 

* TensorFlow Lite and MobileDet
*  Model Conversion
*  Model Benchmarks for MobileDet variants
*  Calibration and Transformation
*  Determining social distance violation
*  Visualization
*  Inference
*  Conclusion

# TensorFlow Lite and MobileDet

Talking of on-device inference, `MobileDet` serves the best. On the COCO object detection task, MobileDets outperform MobileNetV3+SSDLite by 1.7 mAP at comparable mobile CPU inference latencies. MobileDets also outperform MobileNetV2+SSDLite by 1.9 mAP on mobile CPUs. You can know more about [MobileDet from here](https://arxiv.org/abs/2004.14525)

Converting this model to TensorFlow Lite has a purpose as it enables on-device machine learning inference with low latency and a small binary size. TensorFlow Lite is build for developers to run TensorFlow models on mobile, embedded and IoT devices. You can learn more about [TensorFlow Lite from here](https://www.tensorflow.org/lite/guide)

SSD(Single Shot Detector) model takes an image as input. Let's consider, if the input image is of 300x300 pixels, with three channels (red, blue, and green) per pixel, which is then fed to the model as a flattened buffer of 270,000 byte values (300x300x3). If the model is quantized, each value should be a single byte representing a value between 0 and 255. This model outputs 4 arrays(Location, Classes, Confidences, Number of Detection). If we have to convert this model to TensorFlow Lite, we have to first generate the frozen graph that is compatible with the TensorFlow Lite operator set(as explained here - [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) or [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)). The two scripts([TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_ssd_graph.py) and [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_graph_tf2.py)) add optimized postprocessing to the model graph. This model graph is later on quantized to TensorFlow Lite model file with `.tflite` extension with three quantization process.

In the next section, we will see these  three [quantization process](https://www.tensorflow.org/lite/performance/post_training_quantization) : 

1. Dynamic Range quantization
2. Float 16 quantization
3. Full Integer quantization

# Model Conversion 

This is the model [SSD_MobileDet_cpu_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models) which we will quantize ahead. When this model bundle is untar'd, we get following files : pre-trained checkpoints, a TensorFlow Lite (TFLite) compatible model graph, a TFLite model file, a configuration file, and a graph proto. The models were pre-trained on the COCO dataset. `model.ckpt-*` files are the pre-trained checkpoints on the COCO dataset. The `tflite_graph.pb` file is a frozen inference graph that is compatible with the TFLite operator set, which was exported from the pre-trained model checkpoints. `model.tflite` file is a TFLite model that was converted from the `tflite_graph.pb` frozen graph.

```
--- model.ckpt-400000.data-00000-of-00001
--- model.ckpt-400000.index
--- model.ckpt-400000.meta
--- model.tflite
--- pipeline.config
--- tflite_graph.pb
--- tflite_graph.pbtxt
```

### Dynamic Range Quantization

```
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=model_to_be_quantized, 
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

This `dynamic range` quantizes the weights from floating point to integer, which has 8-bits of precision. At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. In the code block above, we need to give the `tflite_graph.pb` file in place of `model_to_be_quantized`. The model accepts the input image to be of  320*320 pixels, so the `input_arrays` and `input_shapes` are set according to that. The `output_arrays` are set to output four arrays : Location of bounding box, Classes of object detected, Confidences, Number of Detections. These are set according to this [guide](https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_ssd_graph.py)

After quantization, we get a model size of 4.9MB.

### Float 16 Quantization

```
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=model_to_be_quantized, 
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
)
converter.allow_custom_ops = True
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

This `float 16` quantization reduces the model size to half with minimal loss in accuracy. It quantizes model weights and bias values from full precision floating point (32-bit) to a reduced precision floating point data type (IEEE FP16). We just have to add one line of code to the previous `dynamic range` code block : `converter.target_spec.supported_types = [tf.float16]`

After quantization, we get a model size of 8.2MB.

### Full Intezer Quantization

```
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=model_to_be_quantized, 
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
)
converter.allow_custom_ops = True
converter.inference_input_type = tf.uint8
converter.quantized_input_stats = {"normalized_input_image_tensor": (128, 128)}
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

# Model Benchmarks for MobileDet variants

Model Benchmark is a way of choosing the best model for your purpose. One way is to know by looking at their `FPS` and `Elapsed Time`. Here are some of the model benchmarks I recorded :

|Model Name|Model Size(MB)|Elapsed Time(s)|FPS|
|--- |--- |--- |--- |
|SSD_mobileDet_cpu_coco_int8|4.9|705.32|0.75|
|SSD_mobileDet_cpu_coco_fp16|8.2|52.79|10.06|
|SSD_mobileDet_cpu_coco_dr|4.9|708.57|0.75|

----------------------

One more way is to use the [TensorFlow Lite Benchmark Tool](https://www.tensorflow.org/lite/performance/measurement). You have to configure [Android Debug Bridge(adb)](https://developer.android.com/studio/command-line/adb) in your laptop and connect it with your android device to use TensorFlow Lite Benchmark Tool and check the inference speed of the model. I have shown only the `fp16` one as this is the fastest among all the three variants.

* The first result in the image given below is with the use of cpu with 4 threads.
* The second result in the image given below is with the use of gpu.

 ![image.png](https://api.wandb.ai/files/cosmo3769/images/projects/206837/234bc2cf.png) 
 
 # Calibration and Transformation
 
 # Determining social distance violation
 
 # Visualization
 
 # Inference

For running TensorFlow Lite model on-device, so that it could make predictions based on input data. This process is 
[Inference](https://www.tensorflow.org/lite/guide/inference). For inference, we need to run it through an interpreter. TensorFlow Lite inference follow these steps given below :

### Load the model

As `SSD_MobileDet_cpu_coco_fp16` showed the best result among all three, we will be going on with loading this model. The `tf.lite.Interpreter` takes in the `.tflite` model file. The tensors are allocated and the model input  shape is defined in `HEIGHT, WIDTH`.

```
tflite_model = "ssd_mobiledet_cpu_coco_fp16.tflite"  
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()
_, HEIGHT, WIDTH, _ = interpreter.get_input_details()[0]['shape']
```

### Set Input Tensor

The code block below will return all the input details of the model.

```
def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
```

### Get Output Tensor

The code block below will return all the output details : `Location of Bounding Box`, `Class, Confidence`, `Number of detection`.

```
def get_output_tensor(interpreter, index):
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor
```

### Pedestrian Detection

```
def pedestrian_detector(interpreter, image, threshold):
  """Returns a list of detection results, each as a tuple of object info."""
  H,W=HEIGHT,WIDTH
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  class_id = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
  
  results = []
  for i in range(count):
    if class_id[i] == 0 and scores[i] >= threshold:
      [ymin,xmin,ymax,xmax]=boxes[i]
      (left, right, top, bottom) = (int(xmin * W), int(xmax * W), int(ymin * H), int(ymax * H))
      area=(right-left+1)*(bottom-top+1)
      if area>=1500:
        continue
      centerX=left+int((right-left)/2)
      centerY=top+int((bottom-top)/2)
      results.append((scores[i],(left,top,right,bottom),(centerX,centerY)))
  return results
```

### Preprocessing Video Frames

```
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocessed_image = frame.resize(
        (
            HEIGHT,
            WIDTH
        ),
        Image.ANTIALIAS)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return preprocessed_image
```

### Output Video Generation Utils

```
def process(video):
  vs=cv2.VideoCapture(video)
  #Capture the first frame of video
  res,image=vs.read()
  if image is None:
    return
  image=cv2.resize(image,(320,320))
  mat=getmap(image)
  fourcc=cv2.VideoWriter_fourcc(*"XVID")
  out=cv2.VideoWriter("result.avi",fourcc,20.0,(320*3,320))
  fps = FPS().start()
  while True:
    res,image=vs.read()
    if image is None:
      break
    #pedestrian detection
    preprocessed_frame = preprocess_frame(image)
    results = pedestrian_detector(interpreter, preprocessed_frame, threshold=0.25)
    preprocessed_frame = np.squeeze(preprocessed_frame) * 255.0
    preprocessed_frame = preprocessed_frame.clip(0, 255)
    preprocessed_frame = preprocessed_frame.squeeze()
    image = np.uint8(preprocessed_frame)
    #calibration
    warped_centroids=calibration(mat, results)
    #Distance-Violation Determination
    violate=calc_dist(warped_centroids)
    #Visualise grid
    grid,warped=visualise_grid(image,mat,warped_centroids,violate)
    #Visualise main frame
    image=visualise_main(image,results,violate)
    #Creating final output frame
    output=cv2.hconcat((image,warped))
    output=cv2.hconcat((output,grid))
    out.write(output)
    fps.update()
  fps.stop()
  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
  # release the file pointers
  print("[INFO] cleaning up...")
  vs.release()
  out.release()
```

# Conclusion
