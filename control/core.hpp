#pragma once

#include <string>
#include <vector>

namespace my_robot {

class RobotController {
public:
    RobotController(const std::string& name);
    std::string getName() const;
    void runInference(const std::vector<float>& input);

private:
    std::string name_;
};

} // namespace my_robot
