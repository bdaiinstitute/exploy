import torch
from my_package import RobotController

def main():
    print("Initializing IsaacLab example...")
    controller = RobotController("IsaacRobot")
    
    # Example Torch tensor to Python list
    input_data = torch.randn(10).tolist()
    controller.run_inference(input_data)
    
    print(f"Controller name: {controller.get_name()}")

if __name__ == "__main__":
    main()
