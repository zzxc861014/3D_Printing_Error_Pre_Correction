import os 
import pymeshlab as ml
import numpy as np
from numba import jit

# path_digital = 'C:/Users/smcmlab-22/Desktop/Pytorch Training/Ground truth_20221017/Digital'
path = 'C:/Users/smcmlab-22/Desktop/Pytorch Training/Ground truth_20221017'
save_path = 'C:/Users/smcmlab-22/Desktop/Position_Encoding_Model'

def file_create(path):
    folder_names = os.listdir(path)
    folder_paths = [os.path.join(path, _) for _ in folder_names]
    file = []
    name = []
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, _) for _ in file_names if _.endswith(".stl")]
        file.append(file_paths)
        name.append(file_names)
    file = np.array(file)
    name = np.array(name)
    return file, name, folder_names


@jit(nopython = "True")
def mass_center_calculate(vertice):
    x = np.mean(vertice[:,0])
    y = np.mean(vertice[:,1])
    z = np.min(vertice[:,2])
    
    return np.array([x,y,z])


def align_point_cloud_to_inertia_axis(vertices):
    vertices = vertices[:,:2]
    # print(vertices)
    covariance_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    angle = -angle # important !!!!

    return angle


def main():
    """
    # Digital
    file_names = os.listdir(path_digital)
    file_paths = [os.path.join(path_digital, _) for _ in file_names if _.endswith(".stl")]
    """
    file_path, file_name, folder_names = file_create(path)
    print(file_path)
    
    for i in range(len(file_path[0,:])):
        for j in range(len(file_path[:,0])):
            input_path = file_path[j][i]
            print(input_path)
            ms = ml.MeshSet()
            ms.load_new_mesh(input_path)

            if input_path.endswith('_D.stl'):
                print(input_path)
                vertex = ms.current_mesh().vertex_matrix()
                mass_center = mass_center_calculate(vertex)
                moved_vector = 0 - mass_center
                moved_vertice = vertex + moved_vector
                print("Mass Center is :", mass_center)
                print("Vector :", moved_vector)
                print("Moved Vertice :", moved_vertice) 

                # Moment of inertia
                
                angle = align_point_cloud_to_inertia_axis(moved_vertice)
                print(np.rad2deg(angle))
            else:   
                print("Vector :", moved_vector)
                print(np.rad2deg(angle))

            ms.compute_matrix_from_translation(traslmethod='XYZ translation', 
                                            axisx=moved_vector[0],
                                            axisy=moved_vector[1],
                                            axisz=moved_vector[2])

            # ms.compute_matrix_by_principal_axis()
            ms.compute_matrix_from_rotation(rotaxis='Z axis', rotcenter='origin', angle=np.rad2deg(angle))

            # Check
            vertex = ms.current_mesh().vertex_matrix()
            if np.any((vertex[:,1] > 0) & (vertex[:,0] > -1) & (vertex[:,0] < 1)):
                ms.compute_matrix_from_rotation(rotaxis='Z axis', rotcenter='origin', angle=180)

            if not os.path.exists(save_path +'/'+ str(folder_names[j])):
                os.makedirs(save_path +'/'+ str(folder_names[j]))

            ms.save_current_mesh(save_path +'/' + str(folder_names[j])+'/' + str(file_name[j][i]))
                       
if __name__ == '__main__':
    main() 