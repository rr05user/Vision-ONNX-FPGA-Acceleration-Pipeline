"""
sitting_standing_with_centroid.py - ONNX Runtime Version

Modified to:
1. Train using TensorFlow (which is ONNX-compatible)
2. Export to ONNX format immediately after training
3. Use ONNX Runtime for inference/testing
4. Avoid Lambda layers and serialization issues
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ONNX conversion imports
import onnx
import onnxruntime as ort
import tf2onnx.convert

print("=" * 70)
print("SITTING/STANDING CLASSIFIER - ONNX RUNTIME VERSION")
print("=" * 70)

# Verify GPU (force CPU only for WSL stability)
print("\nComputing devices available:")
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"  ✓ CPU: {len(cpus)} device(s)")
if gpus:
    print(f"  ⚠️  GPU detected but forced to CPU for stability")
else:
    print("  Using CPU only (stable in WSL)")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "vids")
CSV_PATH = os.path.join(BASE_DIR, "csvPoints.csv")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "sitting_standing_model.onnx")
KERAS_TEMP_PATH = os.path.join(BASE_DIR, "model_temp")

print(f"\n✓ Using local files from: {BASE_DIR}")


def find_videos_locally():
    """Find all videos in local vids folder"""
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Videos directory not found: {VIDEOS_DIR}")
        return [], None
    
    video_files = []
    print(f"Searching for videos in {VIDEOS_DIR}...")
    
    for fname in os.listdir(VIDEOS_DIR):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(VIDEOS_DIR, fname)
            file_size = os.path.getsize(video_path) / (1024*1024)
            video_files.append({
                'name': fname,
                'path': video_path,
                'size': file_size
            })
    
    if video_files:
        print(f"✓ Found {len(video_files)} videos:")
        for v in video_files:
            print(f"  - {v['name']} ({v['size']:.1f} MB)")
    else:
        print(f"❌ No videos found in {VIDEOS_DIR}")
    
    return video_files, VIDEOS_DIR


def filter_sitting_standing_videos(videos):
    """Filter only Sitting and Standing videos"""
    sitting_videos = []
    standing_videos = []
    other_videos = []

    print("\n🔍 Analyzing all videos:")
    for video in videos:
        name = video['name'].lower()

        if 'sit' in name:
            sitting_videos.append(video)
            print(f"  ✓ SITTING: {video['name']}")
        elif 'stand' in name and 'outstanding' not in name:
            standing_videos.append(video)
            print(f"  ✓ STANDING: {video['name']}")
        else:
            other_videos.append(video)
            print(f"  ⊘ SKIP: {video['name']}")

    print(f"\n📊 Video Summary:")
    print(f"  ✓ Sitting videos: {len(sitting_videos)}")
    print(f"  ✓ Standing videos: {len(standing_videos)}")
    print(f"  ⊘ Other videos (skipped): {len(other_videos)}")

    return sitting_videos, standing_videos


print("\n" + "="*70)
print("FINDING VIDEOS")
print("="*70)
videos, folder_id = find_videos_locally()
if videos:
    sitting_videos, standing_videos = filter_sitting_standing_videos(videos)
else:
    print("❌ No videos found! Exiting.")
    exit(1)


def stream_video_from_drive(video_path, video_name):
    """Load video from local file"""
    try:
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        with open(video_path, 'rb') as f:
            video_array = f.read()
        return video_array
    except Exception as e:
        print(f"❌ Error reading {video_name}: {e}")
        return None


def extract_poses_from_stream_with_tracking(video_array, yolo_model, csv_coordinate=None, max_frames=150):
    """Extract YOLO poses from video bytes with CSRT tracking"""
    try:
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_video.mp4")
        with open(temp_path, 'wb') as f:
            f.write(video_array)

        cap = cv2.VideoCapture(temp_path)
        poses = []
        frame_count = 0
        tracked_person_idx = None
        prev_center = None

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, verbose=False)

            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(keypoints) > 0:
                    if tracked_person_idx is None:
                        if csv_coordinate is not None:
                            start_x, start_y = csv_coordinate
                            min_dist = float('inf')
                            for person_idx, (x1, y1, x2, y2) in enumerate(boxes):
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                dist = ((cx - start_x)**2 + (cy - start_y)**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    tracked_person_idx = person_idx
                                    prev_center = (cx, cy)
                        else:
                            tracked_person_idx = 0
                            if len(boxes) > 0:
                                x1, y1, x2, y2 = boxes[0]
                                prev_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    else:
                        if prev_center is not None and len(boxes) > 0:
                            min_dist = float('inf')
                            best_idx = tracked_person_idx
                            for person_idx, (x1, y1, x2, y2) in enumerate(boxes):
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                dist = ((cx - prev_center[0])**2 + (cy - prev_center[1])**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    best_idx = person_idx
                                    prev_center = (cx, cy)
                            tracked_person_idx = best_idx

                    if tracked_person_idx < len(keypoints):
                        person_keypoints = keypoints[tracked_person_idx]
                        poses.append(person_keypoints)

            frame_count += 1

        cap.release()
        os.remove(temp_path)

        if poses:
            return np.array(poses)
        else:
            return None
    except Exception as e:
        print(f"❌ Error extracting poses: {e}")
        return None


def process_sequence(seq):
    """Process pose sequence - EXACT SAME as original"""
    try:
        if len(seq) < 2 or seq.shape[1] != 17:
            return None

        frames = len(seq)
        features = []

        for frame_idx in range(frames):
            keypoints = seq[frame_idx]

            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            if np.all(nose == 0) or np.all(left_hip == 0) or np.all(right_hip == 0):
                features.append(np.zeros(20))
                continue

            frame_features = []
            hip_center = (left_hip + right_hip) / 2
            frame_features.extend([hip_center[0], hip_center[1]])

            shoulder_center = (left_shoulder + right_shoulder) / 2
            frame_features.extend([shoulder_center[0], shoulder_center[1]])

            body_height = np.linalg.norm(shoulder_center - hip_center)
            frame_features.append(body_height)

            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            frame_features.append(shoulder_width)

            hip_width = np.linalg.norm(right_hip - left_hip)
            frame_features.append(hip_width)

            left_wrist = keypoints[9]
            left_arm_length = np.linalg.norm(left_wrist - left_shoulder)
            frame_features.append(left_arm_length)

            right_wrist = keypoints[10]
            right_arm_length = np.linalg.norm(right_wrist - right_shoulder)
            frame_features.append(right_arm_length)

            left_leg_length = np.linalg.norm(left_ankle - left_hip)
            frame_features.append(left_leg_length)

            right_leg_length = np.linalg.norm(right_ankle - right_hip)
            frame_features.append(right_leg_length)

            torso_vector = shoulder_center - hip_center
            torso_angle = np.arctan2(torso_vector[0], torso_vector[1])
            frame_features.append(torso_angle)

            if not np.all(left_knee == 0):
                upper_leg = left_knee - left_hip
                lower_leg = left_ankle - left_knee
                cos_angle = np.dot(upper_leg, lower_leg) / (np.linalg.norm(upper_leg) * np.linalg.norm(lower_leg) + 1e-6)
                left_knee_angle = np.arccos(np.clip(cos_angle, -1, 1))
                frame_features.append(left_knee_angle)
            else:
                frame_features.append(0)

            if not np.all(right_knee == 0):
                upper_leg = right_knee - right_hip
                lower_leg = right_ankle - right_knee
                cos_angle = np.dot(upper_leg, lower_leg) / (np.linalg.norm(upper_leg) * np.linalg.norm(lower_leg) + 1e-6)
                right_knee_angle = np.arccos(np.clip(cos_angle, -1, 1))
                frame_features.append(right_knee_angle)
            else:
                frame_features.append(0)

            hip_nose_dist = abs(nose[1] - hip_center[1])
            frame_features.append(hip_nose_dist)

            left_arm_y_relative = left_wrist[1] - shoulder_center[1]
            right_arm_y_relative = right_wrist[1] - shoulder_center[1]
            frame_features.extend([left_arm_y_relative, right_arm_y_relative])

            def collinearity_score(p1, p2, p3):
                v1 = p2 - p1
                v2 = p3 - p2
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product < 1e-6:
                    return 0.0
                cross = abs(np.cross(v1, v2))
                return cross / norm_product

            left_leg_collinearity = collinearity_score(left_hip, left_knee, left_ankle)
            right_leg_collinearity = collinearity_score(right_hip, right_knee, right_ankle)
            frame_features.extend([left_leg_collinearity, right_leg_collinearity])

            center = (hip_center + shoulder_center) / 2
            extremities = np.array([left_wrist, right_wrist, left_ankle, right_ankle])
            avg_extremity_dist = np.mean([np.linalg.norm(e - center) for e in extremities if not np.all(e == 0)])
            frame_features.append(avg_extremity_dist)

            while len(frame_features) < 20:
                frame_features.append(0)
            frame_features = frame_features[:20]

            features.append(np.array(frame_features[:20], dtype=np.float32))

        if len(features) < 2:
            return None

        features = np.array(features)

        features_normalized = np.zeros_like(features, dtype=np.float32)
        for i in range(features.shape[1]):
            col = features[:, i]
            if np.std(col) > 0:
                features_normalized[:, i] = (col - np.mean(col)) / np.std(col)
            else:
                features_normalized[:, i] = col

        from scipy.ndimage import median_filter
        features_smoothed = median_filter(features_normalized, size=(3, 1))

        return features_smoothed

    except Exception as e:
        print(f"DEBUG: Error in process_sequence: {e}")
        return None


print("\n✓ Processing functions ready!")

# Load YOLO model
print("\n" + "="*70)
print("LOADING YOLO MODEL")
print("="*70)
yolo_model = YOLO('yolo11n-pose.pt')

# Load CSV tracking coordinates
print("\n📋 Loading CSV file with tracking coordinates...")
csv_tracking = {}
try:
    df_tracking = pd.read_csv("csvPoints.csv")
    for idx, row in df_tracking.iterrows():
        try:
            video_id = str(row.iloc[0]).strip()
            activity = str(row.iloc[1]).strip()
            
            points = []
            for col_idx in range(2, 5):
                if col_idx < len(row):
                    point_str = str(row.iloc[col_idx]).strip()
                    if point_str and point_str != 'nan':
                        try:
                            x, y = [int(val.strip()) for val in point_str.replace('(', '').replace(')', '').split(',')]
                            points.append((x, y))
                        except:
                            pass
            
            if points:
                key = (video_id, activity)
                csv_tracking[key] = points
        except:
            pass
    print(f"✓ Loaded tracking coordinates for {len(csv_tracking)} video/activity combinations")
except Exception as e:
    print(f"⚠ Could not load CSV: {e}. Will track first person in each video.")
    csv_tracking = {}

# ============================================================================
# DATA PROCESSING
# ============================================================================

all_poses = []
all_labels = []
all_person_ids = []
failed_videos = []


def extract_person_id_from_filename(filename, point_index=0):
    parts = filename.replace('VID_', '').replace('.mp4', '').split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}_person{point_index}"
    return f"{filename}_person{point_index}"


print("\n" + "="*70)
print("EXTRACTING POSES FROM VIDEOS")
print("="*70)

print("\n🎥 Processing Sitting videos...")
for i, video in enumerate(sitting_videos):
    print(f"[{i+1}/{len(sitting_videos)}] {video['name']}", end=" ... ")

    video_array = stream_video_from_drive(video['path'], video['name'])
    if video_array is None:
        print("❌ Failed to read")
        failed_videos.append(video['name'])
        continue

    video_id_match = None
    for key in csv_tracking:
        if key[0] in video['name'] and key[1].lower() == 'sitting':
            video_id_match = key
            break
    
    csv_coords_list = []
    if video_id_match:
        csv_coords_list = csv_tracking[video_id_match]
        print(f"(found {len(csv_coords_list)} people)", end=" ... ")
    else:
        csv_coords_list = [None]

    num_extracted = 0
    for point_idx, csv_coord in enumerate(csv_coords_list):
        poses = extract_poses_from_stream_with_tracking(video_array, yolo_model, csv_coordinate=csv_coord)
        if poses is not None and len(poses) > 0:
            processed = process_sequence(poses)
            if processed is not None:
                all_poses.append(processed)
                all_labels.append(0)
                person_id = extract_person_id_from_filename(video['name'], point_idx)
                all_person_ids.append(person_id)
                num_extracted += 1
                if point_idx == 0:
                    print(f"(raw shape: {poses.shape})", end=" ... ")
        else:
            if point_idx == 0:
                print(f"⚠ No poses for person {point_idx}", end=" ... ")
    
    if num_extracted > 0:
        print(f"✓ Extracted {num_extracted} person(s)")
    else:
        print("⚠ No poses detected")

print("\n🎥 Processing Standing videos...")
for i, video in enumerate(standing_videos):
    print(f"[{i+1}/{len(standing_videos)}] {video['name']}", end=" ... ")

    video_array = stream_video_from_drive(video['path'], video['name'])
    if video_array is None:
        print("❌ Failed to read")
        failed_videos.append(video['name'])
        continue

    video_id_match = None
    for key in csv_tracking:
        if key[0] in video['name'] and key[1].lower() == 'standing':
            video_id_match = key
            break
    
    csv_coords_list = []
    if video_id_match:
        csv_coords_list = csv_tracking[video_id_match]
        print(f"(found {len(csv_coords_list)} people)", end=" ... ")
    else:
        csv_coords_list = [None]

    num_extracted = 0
    for point_idx, csv_coord in enumerate(csv_coords_list):
        poses = extract_poses_from_stream_with_tracking(video_array, yolo_model, csv_coordinate=csv_coord)
        if poses is not None and len(poses) > 0:
            processed = process_sequence(poses)
            if processed is not None:
                all_poses.append(processed)
                all_labels.append(1)
                person_id = extract_person_id_from_filename(video['name'], point_idx)
                all_person_ids.append(person_id)
                num_extracted += 1
                if point_idx == 0:
                    print(f"(raw shape: {poses.shape})", end=" ... ")
        else:
            if point_idx == 0:
                print(f"⚠ No poses for person {point_idx}", end=" ... ")
    
    if num_extracted > 0:
        print(f"✓ Extracted {num_extracted} person(s)")
    else:
        print("⚠ No poses detected")

print(f"\n✓ Total sequences extracted: {len(all_poses)}")
if failed_videos:
    print(f"⚠ Failed videos: {len(failed_videos)}")

# ============================================================================
# BUILD MODEL - ONNX COMPATIBLE (NO LAMBDA LAYERS)
# ============================================================================

def build_lstm_model(max_len=150, num_features=20, num_classes=2):
    """
    Build LSTM model that is fully ONNX-compatible.
    NO Lambda layers!
    """
    inputs = layers.Input(shape=(max_len, num_features))
    masked = layers.Masking(mask_value=-1.0)(inputs)
    
    x = layers.LayerNormalization()(masked)

    # Three bidirectional LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.LayerNormalization()(x)
    
    # ✓ ONNX-COMPATIBLE POOLING (replaces Lambda layer)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense classifier
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("PREPARING DATA FOR TRAINING")
print("="*70)

max_len = 150
num_features = 20
padded_poses = []
padded_labels = []

print(f"\nExpected shape per sequence: ({max_len}, {num_features})")

skipped = 0
for i, pose in enumerate(all_poses):
    if len(pose.shape) != 2:
        print(f"⚠️ Skipping sequence {i}: unexpected shape {pose.shape}")
        skipped += 1
        continue

    frames, features = pose.shape

    if features != num_features:
        print(f"⚠️ Skipping sequence {i}: has {features} features instead of {num_features}")
        skipped += 1
        continue

    if frames < max_len:
        pad_length = max_len - frames
        padded = np.pad(pose, ((0, pad_length), (0, 0)), 'constant', constant_values=-1)
    else:
        padded = pose[:max_len]

    padded_poses.append(padded)
    padded_labels.append(all_labels[i])

print(f"✓ Prepared {len(padded_poses)} sequences (skipped {skipped})")

if len(padded_poses) == 0:
    print("❌ ERROR: No valid sequences to train on!")
    exit(1)

X = np.array(padded_poses)
y = np.array(padded_labels)

print(f"Data shape before split: {X.shape}")
print(f"Labels shape: {y.shape}")

# ============================================================
# SAVE ONNX INPUTS ONCE (real data, no reruns later)
# ============================================================

# Safety check
if len(X) == 0:
    print("❌ ERROR: X is empty, cannot save samples.")
    exit(1)

# 1) Save ONE real sample (for ONNX sanity checks)
one_sample_path = os.path.join(BASE_DIR, "one_sample_150x20.npy")
np.save(one_sample_path, X[0].astype(np.float32))
print(f"✅ Saved one real ONNX input sample to: {one_sample_path}")
print(f"   Sample shape: {X[0].shape}, dtype: {X[0].dtype}")

# 2) Save CALIBRATION batch (for INT8 quantization later)
CALIB_N = min(200, len(X))
calib_path = os.path.join(BASE_DIR, f"calib_{CALIB_N}_samples.npy")
np.save(calib_path, X[:CALIB_N].astype(np.float32))
print(f"✅ Saved calibration batch to: {calib_path}")
print(f"   Calibration shape: {X[:CALIB_N].shape}")

# ============================================================
# Continue with your normal pipeline
# ============================================================

# Split by person to prevent data leakage
print("\n🔀 Splitting data by PERSON (prevents data leakage)...")

unique_persons = np.array(list(set(all_person_ids)))
print(f"✓ Found {len(unique_persons)} unique people in dataset")

train_persons, test_persons = train_test_split(unique_persons, test_size=0.2, random_state=42)
train_persons, val_persons = train_test_split(train_persons, test_size=0.2, random_state=42)

print(f"  Training people: {len(train_persons)}")
print(f"  Validation people: {len(val_persons)}")
print(f"  Test people: {len(test_persons)}")

train_indices = [i for i, pid in enumerate(all_person_ids) if pid in train_persons]
val_indices = [i for i, pid in enumerate(all_person_ids) if pid in val_persons]
test_indices = [i for i, pid in enumerate(all_person_ids) if pid in test_persons]

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"\n✓ Split complete:")
print(f"  Training set: {len(X_train)} sequences (sitting: {(y_train==0).sum()}, standing: {(y_train==1).sum()})")
print(f"  Validation set: {len(X_val)} sequences (sitting: {(y_val==0).sum()}, standing: {(y_val==1).sum()})")
print(f"  Test set: {len(X_test)} sequences (sitting: {(y_test==0).sum()}, standing: {(y_test==1).sum()})")

# Data augmentation ONLY on training set
print("\n📈 Augmenting TRAINING set only...")
X_train_augmented = [X_train]
y_train_augmented = [y_train]

for aug_idx in range(4):
    X_aug = np.zeros_like(X_train)
    for i, seq in enumerate(X_train):
        aug = seq.copy()
        
        if aug_idx == 0:
            aug = aug + np.random.normal(0, 0.05, aug.shape)
        elif aug_idx == 1:
            aug = aug + np.random.normal(0, 0.1, aug.shape)
        elif aug_idx == 2:
            indices = np.random.choice(np.arange(len(aug)), size=len(aug), replace=True)
            aug = aug[np.sort(indices)]
        else:
            mask = np.random.choice([0.0, 1.0], size=aug.shape, p=[0.1, 0.9])
            aug = aug * mask
        
        X_aug[i] = np.clip(aug, -10, 10)
    
    X_train_augmented.append(X_aug)
    y_train_augmented.append(y_train)

X_train = np.vstack(X_train_augmented)
y_train = np.hstack(y_train_augmented)

print(f"✓ Training set after augmentation: {X_train.shape} (5x increase)")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*70)
print("TRAINING LSTM MODEL")
print("="*70)

model = build_lstm_model()
print(model.summary())

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n⚖️ Class weights: Sitting={class_weights[0]:.2f}, Standing={class_weights[1]:.2f}")

print("\n🧠 Training LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max', min_delta=0.03
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            KERAS_TEMP_PATH, monitor='val_accuracy', save_best_only=True, mode='max'
        )
    ]
)

# ============================================================================
# EVALUATE & SAVE
# ============================================================================

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"📊 Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(f"\n📈 Prediction Breakdown:")
print(f"  Sitting predictions correct: {((y_pred == 0) & (y_test == 0)).sum()}/{(y_test == 0).sum()}")
print(f"  Standing predictions correct: {((y_pred == 1) & (y_test == 1)).sum()}/{(y_test == 1).sum()}")

# ============================================================================
# CONVERT TO ONNX
# ============================================================================

print("\n" + "="*70)
print("CONVERTING TO ONNX FORMAT")
print("="*70)

try:
    # Load the best model
    best_model = tf.keras.models.load_model(KERAS_TEMP_PATH)
    print(f"✓ Loaded best model from {KERAS_TEMP_PATH}")
    
    # Convert to ONNX
    print(f"\n🔄 Converting TensorFlow model to ONNX...")
    spec = (tf.TensorSpec((None, max_len, num_features), tf.float32, name="input"),)
    
    output_path = ONNX_MODEL_PATH
    model_proto, _ = tf2onnx.convert.from_keras(
        best_model,
        input_signature=spec,
        opset=13,
        output_path=output_path
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model validation: SUCCESS")
    
except Exception as e:
    print(f"❌ ONNX conversion failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# TEST WITH ONNX RUNTIME
# ============================================================================

print("\n" + "="*70)
print("TESTING WITH ONNX RUNTIME")
print("="*70)

try:
    # Create ONNX Runtime inference session
    sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    print(f"✓ Created ONNX Runtime inference session")
    
    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"  Input name: {input_name}")
    print(f"  Output name: {output_name}")
    
    # Test inference on test set
    print(f"\nRunning inference on {len(X_test)} test samples...")
    
    onnx_predictions = []
    for i, sample in enumerate(X_test):
        # ONNX expects (batch_size, sequence_length, features)
        input_data = sample[np.newaxis, :, :].astype(np.float32)
        
        # Run inference
        output = sess.run([output_name], {input_name: input_data})[0]
        prediction = np.argmax(output[0])
        onnx_predictions.append(prediction)
    
    onnx_predictions = np.array(onnx_predictions)
    
    # Compare with TensorFlow predictions
    onnx_accuracy = np.mean(onnx_predictions == y_test)
    print(f"\n✅ ONNX Runtime Test Accuracy: {onnx_accuracy*100:.2f}%")
    print(f"   (Should match TensorFlow: {test_accuracy*100:.2f}%)")
    
    # Detailed breakdown
    print(f"\n📈 ONNX Prediction Breakdown:")
    print(f"  Sitting predictions correct: {((onnx_predictions == 0) & (y_test == 0)).sum()}/{(y_test == 0).sum()}")
    print(f"  Standing predictions correct: {((onnx_predictions == 1) & (y_test == 1)).sum()}/{(y_test == 1).sum()}")
    
    if abs(onnx_accuracy - test_accuracy) < 0.01:
        print(f"\n✅ ONNX model matches TensorFlow model!")
    else:
        print(f"\n⚠️ Accuracy difference: {abs(onnx_accuracy - test_accuracy)*100:.2f}%")

except Exception as e:
    print(f"❌ ONNX Runtime inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
✓ Model trained successfully!

📁 Output files:
  • ONNX Model: {ONNX_MODEL_PATH}
  • Keras Model (temporary): {KERAS_TEMP_PATH}

📊 Results:
  • TensorFlow Accuracy: {test_accuracy*100:.2f}%
  • ONNX Runtime Accuracy: {onnx_accuracy*100:.2f}%
  • Test Samples: {len(X_test)}

🔧 Model Details:
  • Input shape: ({max_len}, {num_features})
  • Architecture: 3x Bidirectional LSTM (256→128→64) + GlobalAveragePooling1D + Dense
  • Output: 2 classes (Sitting/Standing)
  • Format: ONNX (framework-agnostic)

✅ The model is now in ONNX format and can be used with:
   • ONNX Runtime (Python, C++, C#, Java)
   • Any framework that supports ONNX
   • Edge devices and mobile platforms
""")

# Optional: Remove temporary keras file
try:
    os.remove(KERAS_TEMP_PATH)
    print(f"🧹 Cleaned up temporary file: {KERAS_TEMP_PATH}")
except:
    pass

print("\n" + "="*70)
print("✅ ALL DONE!")
print("="*70)
