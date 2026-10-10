#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

int main() {
    std::cout << "Testing TT-Metal Simulator MeshDevice Initialization..." << std::endl;
    try {
        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        std::cout << "Successfully created unit MeshDevice on simulator! Arch: " 
                  << static_cast<int>(mesh_device->arch()) << " shape: (" 
                  << mesh_device->shape()[0] << ", " << mesh_device->shape()[1] << ")" << std::endl;
        mesh_device->close();
        std::cout << "Successfully closed MeshDevice!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
