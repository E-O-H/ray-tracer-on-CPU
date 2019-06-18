// C++ include
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "utils.h"

// Shortcut to avoid Eigen:: and std:: everywhere, DO NOT USE IN .h
using namespace std;
using namespace Eigen;

////////// MACRO DEFINES
// #define TBB_PARALLEL  // Uncomment this flag to enable TBB parallel_for
// #define _DEBUG        // Uncomment this flag to output debug information

#ifdef TBB_PARALLEL
#define NOMINMAX
#include "tbb/tbb.h"
#endif

#define ORTHOGRAPHIC 0
#define PERSPECTIVE 1

////////// CONFIG
// Configurable projection parameters
int projection = ORTHOGRAPHIC;    // projection type. 0 for orthographic projection; 1 for perspective projection 

double VIEWPORT_SIZE = 2.0;             // size of the viewport
double PERSPECTIVE_ORIGIN_DISTANCE = 2.0;    // distance of origin of perspective projection

// Configurable rendering parameters
bool renderDiffuse = true;          // toggle diffuse rendering
bool renderSpecular = false;        // toggle specular rendering
bool renderAmbient = false;         // toggle ambient rendering
bool renderShadows = false;         // toggle shadow rendering
bool renderReflection = false;      // toggle reflection rendering

double reflectionCoeff = 0.8;        // default reflection coefficient

unsigned int ambientColorR = 255;   // default ambient color red
unsigned int ambientColorG = 255;   // default ambient color green
unsigned int ambientColorB = 255;   // default ambient color blue
double ambientCoefficient = 0.2;    // default ambient strength

// Other configurable parameters
double floorHeight = -1.0;                   // Height of floor
Vector3i floorColor(255, 255, 255); // Color of floor

unsigned RESOLUTION_X = 800;           // rendering resolution x
unsigned RESOLUTION_Y = 800;           // rendering resolution y

double PRECISION = 1E-3;          // precision for solving linear equations
double EPSILON = 1E-3;            // small displacement used for generating shadow ray

const string dataPath = "./data/";  // path for data files

class Sphere;
class Mesh;
class LightSource;
static vector<Sphere> spheres;            // list to store all spheres
static vector<Mesh> meshes;               // list to store all meshes
static vector<LightSource> lightSources;  // list to store all light sources

// class to represent a light source
class LightSource {
  public:
    double x;
    double y;
    double z;
    unsigned int colorR;
    unsigned int colorG;
    unsigned int colorB;

    LightSource() {};
    LightSource(double _x, double _y, double _z, 
                unsigned int _colorR = 255, 
                unsigned int _colorG = 255, 
                unsigned int _colorB = 255): 
                x(_x), y(_y), z(_z), colorR(_colorR), colorG(_colorG), colorB(_colorB) {};
};

// class to represent a sphere
class Sphere {
  public:
    double x;
    double y;
    double z;
    double r;
    unsigned int colorR;
    unsigned int colorG;
    unsigned int colorB;
    float alpha;
    float diffuse;
    float specular;
    int phongExp;

    Sphere() {};
    Sphere(double _x, double _y, double _z, double _r, 
           unsigned int _colorR = 255, 
           unsigned int _colorG = 255, 
           unsigned int _colorB = 255,
           float _alpha = 1,
           float _diffuse = 1,
           float _specular = 0,
           int _phongExp = 1): 
           x(_x), y(_y), z(_z), r(_r), colorR(_colorR), colorG(_colorG), colorB(_colorB),
           alpha(_alpha), diffuse(_diffuse), specular(_specular), phongExp(_phongExp) {};
};

// Class to represent a mesh
class Mesh {
  public:
    vector<Vector3d>            V;   // List of vertices
    vector<vector<unsigned> >   F;   // List of faces
    
    unsigned int colorR = 255;
    unsigned int colorG = 255;
    unsigned int colorB = 255;
    float alpha = 1;
    float diffuse = 0.8;
    float specular = 300;
    int phongExp = 25;

    Mesh() {};
    
    // set color of the mesh model
    void setColor(unsigned int R, unsigned int G, unsigned int B) {
        colorR = R;
        colorG = G;
        colorB = B;
    }
    
    // zoom the mesh model
    void zoom(double scale) {
        for (Vector3d& vertex : V) {
            vertex *= scale;
        }
    }

    // move the mesh model
    void move(double x, double y, double z) {
        for (Vector3d& vertex : V) {
            vertex += Vector3d(x, y, z);
        }
    }
};

// Function to add a new light source to the scene. Default white light if color is not specified.
void addLightSource(double x, double y, double z, 
                    unsigned int colorR = 255, 
                    unsigned int colorG = 255, 
                    unsigned int colorB = 255) {
    lightSources.push_back(LightSource(x, y, z, colorR, colorG, colorB));
}

// Delete the last added light source.
void deleteLightSource() {
    lightSources.pop_back();
}

// Use this function template to set environment variables
template<class T>
void set(T& var, T val) {
    var = val;
}

// Function to read the spheres data file
vector<Sphere> readSpheres(string filename) {
    try {
        ifstream spheresFile((dataPath + filename).c_str());
        if (!spheresFile.good()) {
            spheresFile.close();
            spheresFile.open(("../" + dataPath + filename).c_str());
        }
        if (!spheresFile.good()) throw 1;

        // Discard the first line (which is instruction on file format)
        spheresFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        unsigned int n;
        spheresFile >> n;
        vector<Sphere> ret(n);
        for (int i = 0; i < n; ++i) {
            spheresFile >> ret[i].x >> ret[i].y >> ret[i].z
            >> ret[i].r >> ret[i].colorR >> ret[i].colorG >> ret[i].colorB
            >> ret[i].alpha >> ret[i].diffuse >> ret[i].specular >> ret[i].phongExp;
        }
        return ret;
    } catch (...) {
        cerr << "Error opening file." << endl;
    }
}

// Function to read a mesh data file
void readMesh(string filename, vector<Mesh>& meshes) {
    try {
        ifstream meshFile((dataPath + filename).c_str());
        if (!meshFile.good()) {
            meshFile.close();
            meshFile.open(("../" + dataPath + filename).c_str());
        }
        if (!meshFile.good()) throw 1;
        
        // Check first line
        string firstLine;
        meshFile >> firstLine;
        if (firstLine != "OFF") throw 1;

        unsigned int nV, nF, nE;
        meshFile >> nV >> nF >> nE;
        Mesh mesh;
        // read vertices
        for (unsigned i = 0; i < nV; ++i) {
            Vector3d vertex;
            meshFile >> vertex.x() >> vertex.y() >> vertex.z();
            mesh.V.push_back(vertex);
        }
        // read faces
        for (unsigned i = 0; i < nF; ++i) {
            unsigned n;
            meshFile >> n;
            vector<unsigned> face;
            for (int j = 0; j < n; ++j) {
                unsigned index;
                meshFile >> index;
                face.push_back(index);
            }
            mesh.F.push_back(face);
        }
        meshes.push_back(mesh);
    } catch (...) {
        cerr << "Error opening file." << endl;
    }
}

// Check if a ray intersects with a triangle.
// Intersection point coodinate is written into ray_intersection (the last argument)
bool intersect_triangle(const Vector3d& ray_origin, const Vector3d& ray_direction, 
                        const Vector3d& triangle_a, const Vector3d& triangle_b, const Vector3d& triangle_c, 
                        Vector3d& ray_intersection) {
    // Construct the Coefficient Matrix of the linear equations
    Matrix3d M_coeff; 
    M_coeff << (triangle_a - triangle_b), (triangle_a - triangle_c) , ray_direction;
    
    // Solve the linear system
    Vector3d solution = M_coeff.colPivHouseholderQr().solve(triangle_a - ray_origin);
    // Check if the solution is valid
    bool solution_exists = (M_coeff * solution).isApprox(triangle_a - ray_origin, PRECISION);
    if (solution_exists 
            && solution[0] + solution[1] <= 1 
            && solution[0] >= 0 
            && solution[1] >= 0
            && solution[2] > 0) {

        // Compute the intersection point
        ray_intersection = ray_origin + ray_direction * solution[2];
        return true;
    }
    return false;
}

// Check if a ray intersects with a sphere.
// Intersection point coodinate is written into ray_intersection (the last argument)
bool intersect_sphere(const Vector3d& ray_origin, const Vector3d& ray_direction, 
                        const Vector3d& center, const double radius, 
                        Vector3d& ray_intersection) {
    double delta = ray_direction.dot(ray_origin - center) 
                    * ray_direction.dot(ray_origin - center) 
                    - ray_direction.squaredNorm()
                      * ((ray_origin - center).squaredNorm() 
                         - radius * radius);
                                
    if (delta >= 0) {
        // The ray hit the sphere, compute the exact intersection point
        double t = ( - sqrt(delta) - ray_direction.dot(ray_origin - center) )
                    / ray_direction.squaredNorm();  // The smaller t is the one closer to camera
        if (t >= 0) {
            ray_intersection = ray_origin + ray_direction * t;
            return true;
        }
    }
    return false;
}

// Color renderer (for one object)
void render_color(const Vector3d& ray_direction, const Vector3d& ray_intersection, const Vector3d& ray_normal, 
                  double colorR, double colorG, double colorB, 
                  float diffuse, float specular, int phongExp, 
                  double& sumLightR, double& sumLightG, double& sumLightB) {
    sumLightR = 0;
    sumLightG = 0;
    sumLightB = 0;
    // render for each light source
    for (unsigned s = 0; s < lightSources.size(); ++s) {
        Vector3d source_origin(lightSources[s].x, lightSources[s].y, lightSources[s].z);
        Vector3d lightColor(lightSources[s].colorR, lightSources[s].colorG, lightSources[s].colorB);

        // Check if this light source is blocked (shadow)
        if (renderShadows) {
            Vector3d shadow_ray_direction = (source_origin - ray_intersection).normalized();
            Vector3d shadow_ray_origin = ray_intersection + shadow_ray_direction * EPSILON;
            Vector3d intersection;
            bool blocked = false;
            for (const Sphere& sphere : spheres) { // check for spheres
                if (intersect_sphere(shadow_ray_origin, shadow_ray_direction, 
                                    Vector3d(sphere.x, sphere.y, sphere.z), sphere.r, intersection)) {
                    if (sphere.alpha > 0.99) {
                        blocked = true;
                        break; // No need checking for other spheres. 
                    }
                    lightColor *= 1 - sphere.alpha; // A very simple alpha model for light through transparent object.
                }
            }
            if (blocked) continue; // Don't render blocked light source.
            for (unsigned m = 0; m < meshes.size(); ++m) { // check for meshes
                for (unsigned f = 0; f < meshes[m].F.size(); ++f) {
                    Vector3d triangle_a, triangle_b, triangle_c;
                    triangle_a = meshes[m].V[meshes[m].F[f][0]];
                    triangle_b = meshes[m].V[meshes[m].F[f][1]];
                    triangle_c = meshes[m].V[meshes[m].F[f][2]];
                    if (intersect_triangle(shadow_ray_origin, shadow_ray_direction, 
                                        triangle_a, triangle_b, triangle_c, intersection)) {
                        if (meshes[m].alpha > 0.99) {
                            blocked = true;
                            break;
                        }
                        lightColor *= 1 - meshes[m].alpha; // A very simple alpha model for light through transparent object.
                        break; // One triangle is enough for the whole mesh.
                    }
                }
                if (blocked) break; // No need to check for other meshes.
            }
            if (blocked) continue; // Don't render blocked light source.
        }
        
        // Render Diffuse
        if (renderDiffuse) {
            double dot_product = 
                (source_origin - ray_intersection).normalized().dot(ray_normal);
            dot_product = max(dot_product, 0.); // Clamp to zero
        
            double diffuseR = dot_product * (colorR / 255.0) * (lightColor[0] / 255.0);
            double diffuseG = dot_product * (colorG / 255.0) * (lightColor[1] / 255.0);
            double diffuseB = dot_product * (colorB / 255.0) * (lightColor[2] / 255.0);
            
            sumLightR += diffuse * diffuseR;
            sumLightG += diffuse * diffuseG;
            sumLightB += diffuse * diffuseB;
        }
        // Render Specular
        if (renderSpecular) {
            double dot_product = 
                ((source_origin - ray_intersection).normalized() - ray_direction).dot(ray_normal) / 2;
            dot_product = max(dot_product, 0.); // Clamp to zero
            double specularR = pow(dot_product, phongExp) 
                                * (colorR / 255.0) * (lightColor[0] / 255.0); 
            double specularG = pow(dot_product, phongExp) 
                                * (colorG / 255.0) * (lightColor[1] / 255.0); 
            double specularB = pow(dot_product, phongExp) 
                                * (colorB / 255.0) * (lightColor[2] / 255.0); 
            sumLightR += specular * specularR;
            sumLightG += specular * specularG;
            sumLightB += specular * specularB;
        }
    }

    // Render Ambient
    if (renderAmbient) {
        sumLightR += ambientCoefficient * (ambientColorR / 255.0) * (colorR / 255.0);
        sumLightG += ambientCoefficient * (ambientColorG / 255.0) * (colorG / 255.0);
        sumLightB += ambientCoefficient * (ambientColorB / 255.0) * (colorB / 255.0);
    }
    // Clamp to one for each object
    // (this only affects objects behind other translucent objects. I found the result to be more interesting when turned off.)
    //sumLightR = min(sumLightR, 1.0);
    //sumLightG = min(sumLightG, 1.0);
    //sumLightB = min(sumLightB, 1.0);
}

// update ray color after going through one (possibly transparent) object
void update_ray_color(float alpha, double sumLightR, double sumLightG, double sumLightB,
                      unsigned i, unsigned j, MatrixXd& C_R, MatrixXd& C_G, MatrixXd& C_B, MatrixXd& A) {                    
    // Add up all render results for one object
    C_R(i, j) = (1 - alpha) * C_R(i, j) + alpha * sumLightR;
    C_G(i, j) = (1 - alpha) * C_G(i, j) + alpha * sumLightG;
    C_B(i, j) = (1 - alpha) * C_B(i, j) + alpha * sumLightB;

    // Disable the alpha mask for this pixel
    A(i,j) = 1;
}

void render_single_ray_no_reflection(const Vector3d& ray_origin, const Vector3d& ray_direction, 
                                     const unsigned& i, const unsigned& j, double& cur_z, 
                                     MatrixXd& C_R, MatrixXd& C_G, MatrixXd& C_B, MatrixXd& A) {
    // Intersect with each of the triangle faces (in each mesh)
    for (unsigned m = 0; m < meshes.size(); ++m) {
        for (unsigned f = 0; f < meshes[m].F.size(); ++f) {
            Vector3d triangle_a, triangle_b, triangle_c; // Vertices of a triangle face
            triangle_a = meshes[m].V[meshes[m].F[f][0]];
            triangle_b = meshes[m].V[meshes[m].F[f][1]];
            triangle_c = meshes[m].V[meshes[m].F[f][2]];

            Vector3d ray_intersection;
            if (intersect_triangle(ray_origin, ray_direction, 
                                    triangle_a, triangle_b, triangle_c, 
                                    ray_intersection)) {
                // check if obscured
                if (cur_z > ray_intersection[2]) {
                    continue;
                }
                // only update cur_z (depth) if object is not transparent
                if (meshes[m].alpha > 0.99) {
                    cur_z = max(cur_z, ray_intersection[2]); // redundent max, just in case i move the code around
                }

                // Compute the normal vector of the face
                Vector3d ray_normal = (triangle_b - triangle_a).cross(triangle_c - triangle_a).normalized();

                // Color rendering
                double sumLightR, sumLightG, sumLightB;
                render_color(ray_direction, ray_intersection, ray_normal, 
                            meshes[m].colorR, meshes[m].colorG, meshes[m].colorB, 
                            meshes[m].diffuse, meshes[m].specular, meshes[m].phongExp, 
                            sumLightR, sumLightG, sumLightB);
                update_ray_color(meshes[m].alpha, sumLightR, sumLightG, sumLightB, 
                                i, j, C_R, C_G, C_B, A);
            }
        } // for-loop end (f
    } // for-loop end (m)

    // Intersect with each of the spheres
    for (unsigned n = 0; n < spheres.size(); ++n) {

        Vector3d center(spheres[n].x, spheres[n].y, spheres[n].z);

        Vector3d ray_intersection;
        if (intersect_sphere(ray_origin, ray_direction, center, spheres[n].r, ray_intersection)) { 

            // check if obscured
            if (cur_z > ray_intersection[2]) {
                continue;
            }
            // only update cur_z (depth) if object is not transparent
            if (spheres[n].alpha > 0.99) {
                cur_z = max(cur_z, ray_intersection[2]); // redundent max, just in case i move the code around
            }

            // Compute normal at the intersection point
            Vector3d ray_normal = (ray_intersection - center).normalized();

            // Color rendering
            double sumLightR, sumLightG, sumLightB;
            render_color(ray_direction, ray_intersection, ray_normal, 
                            spheres[n].colorR, spheres[n].colorG, spheres[n].colorB, 
                            spheres[n].diffuse, spheres[n].specular, spheres[n].phongExp, 
                            sumLightR, sumLightG, sumLightB);
            update_ray_color(spheres[n].alpha, sumLightR, sumLightG, sumLightB, 
                                i, j, C_R, C_G, C_B, A);
        }
    } // for-loop end (n)
}

// (NOTE: Don't render transparent meshes! The current implementation only supports transparency for
// rendering spheres, of which all the transparent ones must be put at the end of the spheres data file, 
// as well as in a sorted order. Non-transparent ones don't need to be sorted.)
void myRayTracer(const string& filename)
{
    MatrixXd C_R = MatrixXd::Zero(RESOLUTION_X, RESOLUTION_Y); // Store the red color
    MatrixXd C_G = MatrixXd::Zero(RESOLUTION_X, RESOLUTION_Y); // Store the green color
    MatrixXd C_B = MatrixXd::Zero(RESOLUTION_X, RESOLUTION_Y); // Store the blue color
    MatrixXd A = MatrixXd::Zero(RESOLUTION_X, RESOLUTION_Y); // Store the alpha mask

    // Calculate camera parameters based on size of viewport
    // Camera is always centered on z-axis, facing -z direction.
    Vector3d origin( - VIEWPORT_SIZE / 2.0, VIEWPORT_SIZE / 2.0, VIEWPORT_SIZE / 2.0);
    Vector3d x_displacement(VIEWPORT_SIZE / RESOLUTION_X, 0, 0);
    Vector3d y_displacement(0, - VIEWPORT_SIZE / RESOLUTION_Y, 0);

    printf("     "); // space to be replaced for displaying progress percentage

    // Loop for each ray
#ifdef TBB_PARALLEL
tbb::parallel_for( unsigned(0), RESOLUTION_X, [&]( unsigned i ) {
    tbb::parallel_for( unsigned(0), RESOLUTION_Y, [&]( unsigned j ) {
#else
    for (unsigned i=0;i<RESOLUTION_X;i++)
    {
        for (unsigned j=0;j<RESOLUTION_Y;j++)
        {
#endif
            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (projection == ORTHOGRAPHIC) {
                ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
                ray_direction = Vector3d(0,0,-1);
            }

            if (projection == PERSPECTIVE) {
               ray_origin = Vector3d(0, 0, PERSPECTIVE_ORIGIN_DISTANCE);
               ray_direction = ((origin + double(i)*x_displacement + double(j)*y_displacement) 
                                - ray_origin).normalized();
            }

            // Record current z of intersection
            double cur_z = numeric_limits<double>::lowest();

            // Intersect with the floor
            if (ray_direction.y()) { // not parallel with the floor
                double t = (floorHeight - ray_origin.y()) / ray_direction.y();
                if (t >= 0) {
                    Vector3d floor_intersection = 
                             ray_origin + ray_direction * (floorHeight - ray_origin.y()) / ray_direction.y();

                    // Calculate reflections from floor
                    if (renderReflection) {
                        // Calculate reflection ray direction
                        Vector3d new_ray_direction = ray_direction - 2 * ray_direction.y() * Vector3d(0, 1, 0);
                        // Cast new reflection ray from the intersection point
                        render_single_ray_no_reflection(floor_intersection, new_ray_direction, i, j, cur_z, 
                                                        C_R, C_G, C_B, A);
                    }

                    // Floor color rendering
                    double sumLightR, sumLightG, sumLightB;
                    render_color(ray_direction, floor_intersection, Vector3d(0, 1, 0), 
                                    floorColor[0], floorColor[1], floorColor[2],  
                                    1.0, 0, 1, sumLightR, sumLightG, sumLightB);
                    update_ray_color(1 - reflectionCoeff , sumLightR, sumLightG, sumLightB, 
                                        i, j, C_R, C_G, C_B, A);
                }
            }

            // Intersect with objects (spheres and meshes)
            render_single_ray_no_reflection(ray_origin, ray_direction, i, j, cur_z, 
                                                            C_R, C_G, C_B, A);
            
#ifdef TBB_PARALLEL
        } ); // TBB parallel_for loop end (j)
    } ); // TBB parallel_for loop end (i)
#else
        } // for-loop end (j)
        // update a progress percentage display after rendering each column
        printf("\b\b\b\b\b%3d%% ", (int) round(i * 100.0 / RESOLUTION_X));
        fflush(stdout);
    } // for-loop end (i)
#endif
    // Save to png
    write_matrix_to_png(C_R, C_G, C_B, A, filename);
}

int main()
{
    spheres = readSpheres("spheres_1.dat");
    
  // task 1 of assignment
    cout << "Rendering task 1... " << flush;

    set(projection, ORTHOGRAPHIC); 
    set(renderDiffuse, true);
    set(renderSpecular, false);
    set(renderAmbient, false);

    addLightSource(-1, 1, 1);

    myRayTracer("task1.png");
    cout << "Done" << endl;

  // task 2 of assignment
    cout << "Rendering task 2... " << flush; 

    set(renderSpecular, true);
    set(renderAmbient, true);
    set(ambientCoefficient, 0.2);

    myRayTracer("task2-1.png");
    addLightSource(2, 1, 1, 100, 200, 255);
    myRayTracer("task2-2.png");
    cout << "Done" << endl;
    
  // task 3 of assignment
    cout << "Rendering task 3... " << flush; 

    set(projection, PERSPECTIVE);
    set(renderSpecular, true);
    set(renderAmbient, true);
    
    deleteLightSource();
    myRayTracer("task3-1.png");
    addLightSource(2, 1, 1, 100, 200, 255);
    myRayTracer("task3-2.png");
    cout << "Done" << endl;

  // task 4 of assignment
    cout << "Rendering task 4... " << flush;

    readMesh("bumpy_cube.off", meshes);
    readMesh("bunny.off", meshes);

    meshes[0].setColor(255, 215, 0); // Set first mesh model to cyan
    meshes[0].zoom(0.2);               // zoom first mesh model
    meshes[0].move(-0.5, 0, -0.5);     // move first mesh model
    
    meshes[1].setColor(255, 100, 255); // Set second mesh model to purple
    meshes[1].zoom(5.0);               // zoom first mesh model
    meshes[1].move(0.5, -1.0, 0.5);    // move first mesh model

    myRayTracer("task4-1.png");
    cout << "Done" << endl;

  // task 5 of assignment
    cout << "Rendering task 5... " << flush;
    
    set(renderShadows, true);
    myRayTracer("task5-1.png");
    cout << "Done" << endl;

  // task 6 of assignment
    cout << "Rendering task 6... " << flush;

    set(renderReflection, true);
    set(floorColor, Vector3i(0, 80, 200));
    myRayTracer("task6-1.png");

    set(reflectionCoeff, 1.0);
    myRayTracer("task6-2.png");

    set(floorColor, Vector3i(255, 255, 255));
    set(reflectionCoeff, 0.5); 
    myRayTracer("task6-3.png");
    cout << "Done" << endl;

  // Additional task: render animation
    cout << "Additional task: Render animation... " << flush;
    spheres = readSpheres("spheres_grid.txt");
    meshes = vector<Mesh>();
    set(renderReflection, false);
    set(lightSources[0].colorR, (unsigned)200);
    set(lightSources[0].colorG, (unsigned)200);
    set(lightSources[0].colorB, (unsigned)200);
    set(lightSources[1].colorR, (unsigned)50);
    set(lightSources[1].colorR, (unsigned)150);
    set(lightSources[1].colorR, (unsigned)255);
    set(lightSources[0].x, 0.0);
    set(lightSources[0].y, 0.0);
    set(lightSources[1].y, 0.0);
    set(lightSources[1].z, 0.0);
    for (int i = 0; i <= 100; ++i) {
        double cameraZ = 1.0 + 0.04 * i;
        set(PERSPECTIVE_ORIGIN_DISTANCE, cameraZ);
        set(lightSources[0].z, -0.2 * i);
        set(lightSources[1].x, 10.0- 0.2 * i);
        myRayTracer("animation_" + to_string(i) + ".png");
    }
    cout << "Done" << endl;
    return 1;
}
