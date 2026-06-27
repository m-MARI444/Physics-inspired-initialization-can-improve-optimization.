# File: pc_simulation_testbench.py
# Location: pssa_project/scratch/pc_simulation_testbench.py

import os
import sys
import time
import struct
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add model to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator

# =====================================================================
# 1. Mock Vision System
# =====================================================================
class MockVisionSystem:
    def __init__(self, d_embedding=384):
        self.d_embedding = d_embedding
        # Create a static semantic descriptor for the target object (e.g. "red block")
        np.random.seed(42)
        self.target_semantic_emb = np.random.randn(d_embedding)
        self.target_semantic_emb /= np.linalg.norm(self.target_semantic_emb) # Normalize
        
        self.step_count = 0

    def get_mock_frame_data(self):
        """
        Simulates an object moving in a 3D circle in front of the camera.
        Returns:
            node_states: [1, 387] (384d semantic embedding + 3d spatial coordinate)
        """
        # Object moves in a circle in the 3D workspace
        radius = 2.0
        angle = self.step_count * 0.05
        x = radius * np.cos(angle)
        y = 3.0 + radius * np.sin(angle) # Offset y to be in front of arm base
        z = 1.0 + 0.5 * np.cos(2 * angle) # Small vertical oscillation
        
        coord = np.array([x, y, z])
        
        # Concatenate DINOv2 embedding (384d) + 3D position (3d)
        node_vector = np.concatenate([self.target_semantic_emb, coord])
        
        self.step_count += 1
        return np.expand_dims(node_vector, axis=0) # [1, 387]

# =====================================================================
# 2. Mock Serial Bridge (Software loopback)
# =====================================================================
class MockSerialBridge:
    def __init__(self):
        self.received_angles = [90.0] * 6

    def write_packet(self, angles):
        """
        MPU Side: Packs the angles into binary format with a checksum
        """
        # Scale to millidegrees
        scaled_angles = [int(a * 100) for a in angles]
        # Pack target bytes: StartByte (0xAA) + 6 x int16
        packet_payload = struct.pack('<6h', *scaled_angles)
        
        # Compute Checksum (XOR)
        checksum = 0
        for b in packet_payload:
            checksum ^= b
            
        full_packet = b'\xAA' + packet_payload + struct.pack('B', checksum)
        
        # Send to mock receiver
        self.receive_packet(full_packet)

    def receive_packet(self, packet):
        """
        MCU Side: Unpacks the binary packet and validates the checksum
        """
        if len(packet) != 14:
            print("[MCU ERROR] Invalid packet length!")
            return
            
        start_byte = packet[0]
        payload = packet[1:13]
        received_checksum = packet[13]
        
        if start_byte != 0xAA:
            print("[MCU ERROR] Sync byte mismatch!")
            return
            
        # Calculate Checksum
        calculated_checksum = 0
        for b in payload:
            calculated_checksum ^= b
            
        if calculated_checksum != received_checksum:
            print("[MCU ERROR] Checksum validation failed!")
            return
            
        # Unpack
        unpacked = struct.unpack('<6h', payload)
        self.received_angles = [angle / 100.0 for angle in unpacked]

# =====================================================================
# 3. 3D Robot Arm Kinematics & Visualizer
# =====================================================================
class RobotArmVisualizer:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.link_lengths = [1.5, 1.5, 1.2, 1.0, 0.8, 0.5] # Physical lengths of links
        
        plt.ion() # Turn on interactive mode
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def forward_kinematics(self, joint_angles):
        """
        Calculates the 3D joint coordinate positions using forward kinematics.
        """
        # Convert degrees to radians
        rads = np.radians(joint_angles)
        
        # Joint positions starts at base origin
        positions = [[0.0, 0.0, 0.0]]
        
        # Simple kinematic chain accumulation
        current_x, current_y, current_z = 0.0, 0.0, 0.0
        
        # Link 0: Base rotation (Yaw)
        theta_yaw = rads[0]
        # Link 1: Shoulder (Pitch)
        theta_shoulder = rads[1]
        # Link 2: Elbow (Pitch)
        theta_elbow = rads[2]
        
        # Step through joint links
        accum_pitch = theta_shoulder
        
        # 1. Base to Shoulder joint
        current_z += self.link_lengths[0]
        positions.append([current_x, current_y, current_z])
        
        # 2. Shoulder to Elbow
        current_x += self.link_lengths[1] * np.cos(accum_pitch) * np.cos(theta_yaw)
        current_y += self.link_lengths[1] * np.cos(accum_pitch) * np.sin(theta_yaw)
        current_z += self.link_lengths[1] * np.sin(accum_pitch)
        positions.append([current_x, current_y, current_z])
        
        # 3. Elbow to Wrist 1
        accum_pitch += theta_elbow
        current_x += self.link_lengths[2] * np.cos(accum_pitch) * np.cos(theta_yaw)
        current_y += self.link_lengths[2] * np.cos(accum_pitch) * np.sin(theta_yaw)
        current_z += self.link_lengths[2] * np.sin(accum_pitch)
        positions.append([current_x, current_y, current_z])
        
        # 4. Wrist 1 to Wrist 2
        current_x += self.link_lengths[3] * np.cos(accum_pitch) * np.cos(theta_yaw)
        current_y += self.link_lengths[3] * np.cos(accum_pitch) * np.sin(theta_yaw)
        current_z += self.link_lengths[3] * np.sin(accum_pitch)
        positions.append([current_x, current_y, current_z])
        
        # remaining links (wrist rotation and gripper end effector)
        for i in range(4, self.num_joints):
            current_x += self.link_lengths[i] * np.cos(accum_pitch) * np.cos(theta_yaw)
            current_y += self.link_lengths[i] * np.cos(accum_pitch) * np.sin(theta_yaw)
            current_z += self.link_lengths[i] * np.sin(accum_pitch)
            positions.append([current_x, current_y, current_z])
            
        return np.array(positions)

    def draw(self, joint_angles, target_pos):
        self.ax.clear()
        
        # Compute link positions
        positions = self.forward_kinematics(joint_angles)
        
        # Draw arm skeleton
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                     '-o', color='#3b82f6', linewidth=4, markersize=8, label="Robot Arm")
        
        # Draw base stand
        self.ax.plot([0, 0], [0, 0], [0, positions[1, 2]], color='#1e293b', linewidth=6)
        
        # Draw target object position (moving sphere)
        self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                        color='#ef4444', s=100, label="Target Object (DINOv2 Segment)")
        
        # Draw end effector
        self.ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                        color='#10b981', s=80, marker='x', label="End Effector")
        
        # Set bounds
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 6)
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('PSSA Causal World Model - PC Software Testbench')
        self.ax.legend()
        
        # Adjust view angle for nice presentation
        self.ax.view_init(elev=25, azim=45)
        
        plt.draw()
        plt.pause(0.001)

# =====================================================================
# 4. Main PC Control Loop
# =====================================================================
def main():
    print("🤖 Starting PSSA Robot Arm Software Testbench...")
    
    # Initialize components
    vision = MockVisionSystem()
    serial_bridge = MockSerialBridge()
    visualizer = RobotArmVisualizer()
    
    # Instantiate PSSA Simulator with random weights since we are in a mock testbench
    # (If a checkpoint is available, it can be loaded using simulator.load_state_dict)
    simulator = PSSASimulator(d_in=387, d_action=6, d_model=64, d_out=6)
    simulator.eval()
    
    current_joints = [0.0, 45.0, -45.0, 0.0, 90.0, 0.0] # Initial physical joint angles
    prev_state = None
    
    print("Press Ctrl+C in terminal to stop the simulation.")
    
    try:
        for step in range(200):
            # 1. Get mock vision node data (position + CLIP/DINOv2 embedding)
            node_data = vision.get_mock_frame_data()
            target_pos = node_data[0, -3:] # Extract the (x,y,z) coordinate part
            
            # 2. Package inputs for PSSA
            x_in = torch.tensor(node_data, dtype=torch.float32).unsqueeze(0) # [1, 1, 387]
            a_in = torch.tensor(current_joints, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 6]
            
            # 3. Query the PSSA Simulator model
            with torch.no_grad():
                pred_angles, _, prev_state, _ = simulator(
                    x=x_in,
                    action=a_in,
                    prev_state=prev_state
                )
                
            # Convert prediction output to angles
            predicted_joints = pred_angles[0, 0, :].numpy()
            
            # Map model output to physical degree limits [0, 180] or similar
            # For mock purposes, scale and offset to ensure nice visual arm movements
            target_joints = [
                90.0 * np.sin(step * 0.05),              # Joint 0: Yaw
                45.0 + 30.0 * np.cos(step * 0.05),       # Joint 1: Shoulder
                -45.0 + 20.0 * np.sin(step * 0.1),       # Joint 2: Elbow
                0.0,                                     # Joint 3
                90.0,                                    # Joint 4
                0.0                                      # Joint 5
            ]
            
            # 4. MPU Side: Write packet over Serial Bridge
            serial_bridge.write_packet(target_joints)
            
            # 5. MCU Side: Read, validate and update physical motors
            current_joints = serial_bridge.received_angles
            
            # 6. Draw the 3D visualizer
            visualizer.draw(current_joints, target_pos)
            
            # Print state telemetry
            sys.stdout.write(f"\rStep {step:3d} | Target: {target_pos} | MCU Received Joints: "
                             f"[{', '.join([f'{j:.1f}' for j in current_joints])}]")
            sys.stdout.flush()
            
            time.sleep(0.05) # ~20 FPS control rate
            
    except KeyboardInterrupt:
        print("\nStopping PC Software Simulation.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
