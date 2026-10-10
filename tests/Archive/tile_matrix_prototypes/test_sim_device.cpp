#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

int main() {
    std::cout << "Testing TT-Metal Simulator Device Initialization..." << std::endl;
    try {
        auto* device = tt::tt_metal::CreateDevice(0);
        std::cout << "Successfully created device 0 on simulator! Arch: " 
                  << static_cast<int>(device->arch()) << std::endl;
        tt::tt_metal::CloseDevice(device);
        std::cout << "Successfully closed device 0!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
