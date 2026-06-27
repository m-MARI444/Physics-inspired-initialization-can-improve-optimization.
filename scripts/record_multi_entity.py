import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import torch
import time
import os

class MultiEntityRecorder(Node):
    def __init__(self, cube_frame_name="cube"):
        super().__init__('multi_entity_recorder')
        
        self.cube_frame_name = cube_frame_name
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
            
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
            
        self.arm_states = []
        self.cube_states = []
        
        self.current_arm = None
        self.current_cube = [0.0, 0.0, 0.0] # Default origin if not found yet
        
        self.is_recording = False
        self.timer = self.create_timer(0.05, self.record_step) # 20 Hz
        
        print(f"Waiting for /joint_states and TF frame '{cube_frame_name}'...")
        print("Press Enter in another terminal to start recording, then Ctrl+C to save.")

    def joint_callback(self, msg):
        # Assuming the SO-ARM100 has 6 joints
        self.current_arm = list(msg.position)
        
    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == self.cube_frame_name:
                t = transform.transform.translation
                self.current_cube = [t.x, t.y, t.z]

    def record_step(self):
        if self.is_recording and self.current_arm is not None:
            self.arm_states.append(self.current_arm)
            self.cube_states.append(self.current_cube)

def main(args=None):
    rclpy.init(args=args)
    recorder = MultiEntityRecorder(cube_frame_name="box") # Change to whatever your cube is named
    
    input("Press ENTER to start recording...")
    print("Recording started. Move the arm to push the cube! Press Ctrl+C to stop and save.")
    recorder.is_recording = True
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nSaved {len(recorder.arm_states)} timesteps.")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        torch.save(torch.tensor(recorder.arm_states), os.path.join(data_dir, "entity_arm_states.pt"))
        torch.save(torch.tensor(recorder.cube_states), os.path.join(data_dir, "entity_cube_states.pt"))
        
        print("Data saved to data/entity_arm_states.pt and data/entity_cube_states.pt")
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
