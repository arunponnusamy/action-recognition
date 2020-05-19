# import necessary packages
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import cv2
import os

# parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--video', '-v', required=True,
                help='path to video clip')
ap.add_argument('--label', '-l', required=True,
                help='path to label file')
ap.add_argument('--max_frames', '-m', type=int, default=0,
                help='max no. of frames to use from video clip')
args = ap.parse_args()


# center crop the frame for resizing
def center_crop(frame):

    h, w = frame.shape[:2]
    min_dim = min(h, w)

    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)

    crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    return crop


# load video clip from disk and preprocess the frames
def load_video(video_path, max_frames=0, resize=(224,224)):
    
    clip = cv2.VideoCapture(video_path)
    frames = []

    if not clip.isOpened():
        print('[ERROR] Could not open video')
        exit()
        
    while clip.isOpened():

        status, frame = clip.read()

        if not status:
            break

        frame = center_crop(frame)
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) == max_frames:
            break

    clip.release()

    return np.array(frames) / 255.0


# read labels from file
def read_label(label_file):

    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    return labels    


# load video clip
frames = load_video(args.video)
frames = tf.constant(frames, dtype=tf.float32)[tf.newaxis, ...]

# load tf hub module
module = "https://tfhub.dev/deepmind/i3d-kinetics-600/1"
i3d = hub.load(module).signatures['default']

# run inference
logits = i3d(frames)['default'][0]
scores = tf.nn.softmax(logits).numpy()

# read labels
labels = read_label(args.label)

# print top 5 predictions
for i in np.argsort(scores)[::-1][:5]:
    print(labels[i], scores[i])

