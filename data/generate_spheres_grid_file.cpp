// Used to generate spheres grid data file for demonstration
#include <fstream>
using namespace std;
int main() {
    std::ofstream file("spheres_grid.txt");
    file << "// format: x y z radius color_R color_G color_B alpha diffuse_coefficient specular_coefficient phong_exponent (please don't delete this line)" << endl;
    file << 1000 << endl;
    for (int i = 0; i < 10; ++i) {
        double x = -9.0 + 2.0 * i;
        for (int j = 0; j < 10; ++j) {
            double y = -9.0 + 2.0 * j;
            for (int k = 0; k < 10; ++k) {
                double z = -2.0 * k;
                file << x << " " << y << " " << z << " ";
                file << "0.3 255 255 255 1 0.5 300 50" << endl;
            }
        }
    }
    file.close();
}
