import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class FootballPlayerTracker:
    def __init__(self, video_path, output_csv="./assets/csv/player_positions_1.csv"):
        """
        Initialize the player tracker
        
        Args:
            video_path: Path to the football video file
            output_csv: Path to save the tracking data
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = None
        self.tracking_data = []
        
        # Initialize YOLOv8 model (using pre-trained model for person detection)
        self.model = YOLO('yolov8n.pt')  # You can use 'yolov8m.pt' or 'yolov8l.pt' for better accuracy
        
        # Define pitch dimensions (in pixels - will be calibrated from video)
        self.pitch_length = 105  # meters
        self.pitch_width = 68    # meters
        
    def calibrate_pitch(self, frame):
        """
        If you know specific points on the pitch, you can calibrate here.
        For now, we'll work in pixel coordinates.
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
    def detect_and_track(self, output_video_path=None, show_video=False):
        """
        Main function to process video and track players
        
        Args:
            output_video_path: Path to save annotated video (optional)
            show_video: Whether to display the video during processing
        """
        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer for output (if needed)
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                                 (int(self.cap.get(3)), int(self.cap.get(4))))
        
        frame_count = 0
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Run YOLOv8 tracking on the frame
            # 'persist=True' maintains track IDs across frames
            results = self.model.track(frame, persist=True, 
                                      classes=[0, 32],  # Class 0 is 'person' in COCO and Class 32 is 'ball'
                                      conf=0.3,     # Confidence threshold
                                      iou=0.5,      # IoU threshold for NMS
                                      tracker="bytetrack.yaml")  # Using ByteTrack tracker
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Extract tracking data if detections exist
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
                track_ids = results[0].boxes.id.cpu().numpy()  # Track IDs
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidences
                
                for box, track_id, confidence in zip(boxes, track_ids, confidences):
                    # Calculate center of bounding box
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    
                    # Width and height of bounding box
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    
                    # Store tracking data
                    self.tracking_data.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'track_id': int(track_id),
                        'x_pixel': float(x_center),
                        'y_pixel': float(y_center),
                        'x_normalized': float(x_center / self.cap.get(3)),
                        'y_normalized': float(y_center / self.cap.get(4)),
                        'width': float(width),
                        'height': float(height),
                        'confidence': float(confidence)
                    })
                    
                    # Add text to annotated frame
                    cv2.putText(annotated_frame, f'ID: {int(track_id)}', 
                               (int(box[0]), int(box[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save or display video
            if output_video_path:
                out.write(annotated_frame)
            
            if show_video:
                cv2.imshow('Player Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Clean up
        self.cap.release()
        if output_video_path:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete. Tracked {frame_count} frames.")
        
        # Save tracking data to CSV
        self.save_tracking_data()
        
        return self.tracking_data
    
    def save_tracking_data(self):
        """Save tracking data to CSV file"""
        if not self.tracking_data:
            print("No tracking data to save.")
            return
        
        df = pd.DataFrame(self.tracking_data)
        df.to_csv(self.output_csv, index=False)
        print(f"Tracking data saved to {self.output_csv}")
        
        # Print summary statistics
        unique_players = df['track_id'].nunique()
        print(f"Unique players tracked: {unique_players}")
        print(f"Total detections: {len(df)}")
        
    def visualize_tracking(self, frame_number=None):
        """
        Visualize player positions for a specific frame or all frames
        
        Args:
            frame_number: Specific frame to visualize (if None, shows all frames)
        """
        if not self.tracking_data:
            print("No tracking data available. Run detect_and_track() first.")
            return
        
        df = pd.DataFrame(self.tracking_data)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if frame_number is not None:
            # Plot specific frame
            frame_data = df[df['frame'] == frame_number]
            ax = axes[0]
            ax.scatter(frame_data['x_pixel'], frame_data['y_pixel'], 
                      c=frame_data['track_id'], cmap='tab20', s=100)
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title(f'Player Positions - Frame {frame_number}')
            ax.invert_yaxis()  # Invert y-axis to match image coordinates
            
            # Add player IDs as annotations
            for _, row in frame_data.iterrows():
                ax.annotate(f"ID:{int(row['track_id'])}", 
                          (row['x_pixel'], row['y_pixel']),
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=8)
        else:
            # Plot all frames with color by track_id
            unique_ids = df['track_id'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
            
            ax = axes[0]
            for track_id, color in zip(unique_ids, colors):
                player_data = df[df['track_id'] == track_id]
                ax.scatter(player_data['x_pixel'], player_data['y_pixel'], 
                          c=[color], s=10, alpha=0.5, label=f'ID: {int(track_id)}')
            
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title('Player Trajectories - All Frames')
            ax.invert_yaxis()
        
        # Plot player trajectories over time
        ax2 = axes[1]
        unique_ids = df['track_id'].unique()[:10]  # Limit to first 10 players for clarity
        
        for track_id in unique_ids:
            player_data = df[df['track_id'] == track_id]
            ax2.plot(player_data['timestamp'], player_data['x_pixel'], 
                    label=f'ID: {int(track_id)}', alpha=0.7)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('X Position (pixels)')
        ax2.set_title('Player Movement Over Time (X-axis)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def calculate_metrics(self):
        """Calculate basic spatial metrics from tracking data"""
        if not self.tracking_data:
            print("No tracking data available.")
            return
        
        df = pd.DataFrame(self.tracking_data)
        metrics = {}
        
        # Team centroids (assuming we could separate teams by position or color)
        # For now, calculate overall centroid
        metrics['overall_centroid_x'] = df.groupby('frame')['x_pixel'].mean()
        metrics['overall_centroid_y'] = df.groupby('frame')['y_pixel'].mean()
        
        # Team spread (standard deviation)
        metrics['spread_x'] = df.groupby('frame')['x_pixel'].std()
        metrics['spread_y'] = df.groupby('frame')['y_pixel'].std()
        
        # Player movement speed (if temporal data available)
        # This requires frame rate information
        df['velocity_x'] = df.groupby('track_id')['x_pixel'].diff()
        df['velocity_y'] = df.groupby('track_id')['y_pixel'].diff()
        df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
        
        return df, metrics

# Main execution
if __name__ == "__main__":
    # Initialize tracker
    video_path = "./assets/input/input2.mp4"  # Change this to your video path
    tracker = FootballPlayerTracker(video_path, output_csv="./assets/csv/player_positions_1.csv")
    
    # Run detection and tracking
    # Set show_video=True to see real-time tracking
    tracking_data = tracker.detect_and_track(
        output_video_path="./assets/output/output3.mp4",  # Optional: save annotated video
        show_video=False  # Set to True to display while processing
    )
    
    # Visualize results
    tracker.visualize_tracking(frame_number=100)  # Visualize frame 100
    tracker.visualize_tracking()  # Visualize all frames
    
    # Calculate metrics
    df, metrics = tracker.calculate_metrics()
    
    # Print some statistics
    print("\n=== Tracking Statistics ===")
    print(f"Total frames processed: {df['frame'].max()}")
    print(f"Unique players tracked: {df['track_id'].nunique()}")
    print(f"Average players per frame: {df.groupby('frame').size().mean():.2f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("./assets/csv/spatial_metrics_1.csv")
    print("Spatial metrics saved to spatial_metrics.csv")