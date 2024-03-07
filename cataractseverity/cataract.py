from camconnect import PredictImage
import tensorflow as tf
import cv2
import numpy as np

def inference_model(frame):
    CLASS_NAMES = ['immature', 'mature', 'normal']

    # Load TFLite Model and Allocate Tensors.
    interpreter_segmentation = tf.lite.Interpreter(model_path="model1.tflite")
    interpreter_segmentation.allocate_tensors()
    interpreter_classification = tf.lite.Interpreter(model_path="model2.tflite")
    interpreter_classification.allocate_tensors()

    # Get Height and Width.
    _, height, width, _ = interpreter_segmentation.get_input_details()[0]['shape']

    # Load Image from Path and Resize
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (width, height)).astype(np.float32)
    image = image / 255.0

    # Add Batch Dimension
    input_image = np.expand_dims(image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    # Set Input and Invoke Segmentation Model
    interpreter_segmentation.set_tensor(interpreter_segmentation.get_input_details()[0]['index'], input_image)
    interpreter_segmentation.invoke()

    # Get Output
    output_segmentation = np.squeeze(interpreter_segmentation.get_tensor(interpreter_segmentation.get_output_details()[0]['index']))
    output_segmentation = np.where(output_segmentation > 0.6, 1, 0).astype(np.uint8)

    # Fuse Image with Segmentation Mask with Bitwise Operation
    multipy_image = cv2.resize(frame, (width, height))
    multipy_image = cv2.bitwise_and(multipy_image, multipy_image, mask=output_segmentation)
    multipy_image = cv2.cvtColor(multipy_image, cv2.COLOR_BGR2RGB)

    # Add Batch Dimension for Predicted Image
    input_image = np.expand_dims(multipy_image, axis=0).astype(np.float32)

    # Set Input and Invoke Classification Model
    interpreter_classification.set_tensor(interpreter_classification.get_input_details()[0]['index'], input_image)
    interpreter_classification.invoke()

    # Get Output
    output_classification = np.squeeze(interpreter_classification.get_tensor(interpreter_classification.get_output_details()[0]['index']))
    output_classification = CLASS_NAMES[np.argmax(output_classification)]

    return output_classification

def run():
    PredictImage(cv2.VideoCapture(0), width=480, height=320, save_image=True, window_name="Predict Image", inference_model=inference_model).run()