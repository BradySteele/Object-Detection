import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

model_handle = 'https://tfhub.dev/tensorflow/efficientdet/d0/1'
model = hub.load(model_handle)

infer = model.signatures['serving_default']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (512, 512))
    input_tensor = np.expand_dims(resized_frame, axis=0)

    detections = model(input_tensor)

    for i in range(len(detections['detection_boxes'][0])):
        score = detections['detection_scores'][0, i].numpy()
        label = int(detections['detection_classes'][0, i].numpy())
        bbox = detections['detection_boxes'][0, i].numpy()

        if label == 1 and score > 0.5: 
            h, w, _ = frame.shape
            ymin, xmin, ymax, xmax = bbox
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()