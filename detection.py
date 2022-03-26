import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
import mediapipe as mp
import six
import time


WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
final_text = ''

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resH = hands.process(imgRGB)
    black = np.zeros(frame.shape, dtype=np.uint8)
    if resH.multi_hand_landmarks:
        for handLms in resH.multi_hand_landmarks:
            mpDraw.draw_landmarks(black, handLms, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=1))
    cv2.rectangle(frame, (int(0.5 * frame.shape[1]), 100), (frame.shape[1], int(0.8 * frame.shape[0])), (255, 0, 0), 2)
    cv2.rectangle(black, (int(0.5 * frame.shape[1]), 100), (frame.shape[1], int(0.8 * frame.shape[0])), (255, 0, 0), 2)
    roi = black[100:int(0.8 * frame.shape[0]), int(0.5 * frame.shape[1]):frame.shape[1]]

    image_np = np.array(roi)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.5,
        agnostic_mode=False)
    final = frame.copy() 
    
    classes = detections['detection_classes'] + label_id_offset
    
    #print(detections['detection_boxes'].shape[0])
    
    for i in range(detections['detection_boxes'].shape[0]):
        if detections['detection_scores'] is None or detections['detection_scores'][i] > 0.85:
            class_name = category_index[classes[i]]['name']
            final_text += str(class_name)
            time.sleep(1)
            
    #print(final_text)        
    cv2.putText(final, final_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    cv2.imshow('Res', cv2.resize(final, (800, 600)))
    
    k = cv2.waitKey(1)
    
    if k == ord('q'):  # Bam q de thoat
        cv2.destroyWindow('Alphabet')
        cv2.destroyWindow('object detection')
        cv2.destroyWindow('Res')
        break
        
    if k == ord('c'):
        final_text = ''
    """
    if cv2.waitKey(1) & 0xFF == ord('c'):
        final_text = ''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow('Alphabet')
        cv2.destroyWindow('object detection')
        cv2.destroyWindow('Res')
        #cap.release()
        break
    """