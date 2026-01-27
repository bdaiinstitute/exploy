#include "core.hpp"
#include <iostream>

int main() {
    std::cout << "Initializing ROS2 Control example..." << std::endl;
    my_robot::RobotController controller("ROS2Robot");
    
    std::vector<float> input = {1.0, 2.0, 3.0};
    controller.run_inference(input);
    
    std::cout << "Controller name: " << controller.getName() << std::endl;
    return 0;
}
