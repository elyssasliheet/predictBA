import numpy as np
import pandas as pd
import shutil
import os 
import glob
import sys

def get_features(p, L):    

    test_set_comps = pd.read_csv("test_set_comps_final_for_CNN.txt", header=None, names = ['PDB_IDs'])
    train_set_comps = pd.read_csv("train_set_comps_final_for_CNN.txt", header=None, names = ['PDB_IDs'])

    test_set_comps['FilePathComplex'] = '/users/esliheet/esliheet/predictBA/v2007/' + test_set_comps['PDB_IDs'] + "/" + test_set_comps['PDB_IDs'] +'_complex.pqr'
    train_set_comps['FilePathComplex'] = '/users/esliheet/esliheet/predictBA/v2007/' + train_set_comps['PDB_IDs'] + "/" + train_set_comps['PDB_IDs'] + '_complex.pqr'

    protein_test_paths = []
    ligand_test_paths = []
    for comp_file in test_set_comps['FilePathComplex']:
        pro_file = get_protein(comp_file)
        protein_test_paths.append(pro_file)
        ligand_file = get_ligand(comp_file)
        ligand_test_paths.append(ligand_file)

    protein_train_paths = []
    ligand_train_paths = []
    for comp_file in train_set_comps['FilePathComplex']:
        pro_file = get_protein(comp_file)
        protein_train_paths.append(pro_file)
        ligand_file = get_ligand(comp_file)
        ligand_train_paths.append(ligand_file)
    
    test_set_comps['FilePathProtein'] = protein_test_paths
    test_set_comps['FilePathLigand'] = ligand_test_paths

    train_set_comps['FilePathProtein'] = protein_train_paths
    train_set_comps['FilePathLigand'] = ligand_train_paths


    #y_df = pd.concat([test_set_comps, train_set_comps], axis = 0)
    #print(y_df.shape)
    #y_df = y_df.drop_duplicates(subset=['PDB_IDs']).reset_index(drop=True)
    #print(y_df.shape)

    # Make list of file paths and protein IDs 
    test_set_comps["e_features_commands_complex"] = './a.out ' + test_set_comps["FilePathComplex"] +  ' 0.0 ' + str(p) + ' ' + str(L)
    train_set_comps["e_features_commands_complex"] = './a.out ' + train_set_comps["FilePathComplex"] +  ' 0.0 ' + str(p) + ' ' + str(L)


    # Make list of file paths and protein IDs 
    test_set_comps["e_features_commands_protein"] = './a.out ' + test_set_comps["FilePathProtein"] +  ' 0.0 ' + str(p) + ' ' + str(L)
    train_set_comps["e_features_commands_protein"] = './a.out ' + train_set_comps["FilePathProtein"] +  ' 0.0 ' + str(p) + ' ' + str(L)


    # Make list of file paths and protein IDs 
    test_set_comps["e_features_commands_ligand"] = './a.out ' + test_set_comps["FilePathLigand"] +  ' 0.0 ' + str(p) + ' ' + str(L)
    train_set_comps["e_features_commands_ligand"] = './a.out ' + train_set_comps["FilePathLigand"] +  ' 0.0 ' + str(p) + ' ' + str(L)


    test_features = []
    dir = "/users/esliheet/esliheet/e_features"
    os.chdir(dir)
    os.system('make clean')
    os.system('make')
    for command1, command2, command3 in zip(test_set_comps["e_features_commands_complex"], test_set_comps["e_features_commands_protein"], test_set_comps["e_features_commands_ligand"]):
        print("========================")
        print(command1)
        # complex command
        os.system(command1)
        feature_complex  = np.loadtxt('efeature.txt')
        print("feature_complex")
        print(feature_complex)

        # protein command
        os.system(command2)
        feature_protein  = np.loadtxt('efeature.txt')
        print("feature_protein")
        print(feature_protein)

        # ligand command
        os.system(command3)
        feature_ligand  = np.loadtxt('efeature.txt')
        print("feature_ligand")
        print(feature_ligand)

        # final feature for binding affinity prediction
        final_feature = feature_complex - feature_protein - feature_ligand
        print("final_feature")
        print(final_feature)
        test_features.append(final_feature)    
    print("+===============================================")
    os.system('make clean')
    dir = "/users/esliheet/esliheet/predictBA"
    os.chdir(dir)


    train_features = []
    dir = "/users/esliheet/esliheet/e_features"
    os.chdir(dir)
    os.system('make clean')
    os.system('make')
    for command1, command2, command3 in zip(train_set_comps["e_features_commands_complex"], train_set_comps["e_features_commands_protein"], train_set_comps["e_features_commands_ligand"]):
        # complex command
        os.system(command1)
        feature_complex  = np.loadtxt('efeature.txt')
        print("feature_complex")
        print(feature_complex)

        # protein command
        os.system(command2)
        feature_protein  = np.loadtxt('efeature.txt')
        print("feature_protein")
        print(feature_protein)

        # ligand command
        os.system(command3)
        feature_ligand  = np.loadtxt('efeature.txt')
        print("feature_ligand")
        print(feature_ligand)

        # final feature for binding affinity prediction
        final_feature = feature_complex - feature_protein - feature_ligand
        print("final_feature")
        print(final_feature)
        train_features.append(final_feature)    
    os.system('make clean')
    dir = "/users/esliheet/esliheet/predictBA"
    os.chdir(dir)

    # Convert features to numpy array
    X_test_electrostatic = np.vstack(test_features)
    X_test_electrostatic_df = pd.DataFrame(X_test_electrostatic)  
    X_test_electrostatic_df.reset_index(drop=True, inplace=True)
    print("X_test_electrostatic_df shape: ", X_test_electrostatic_df.shape)

    # Convert features to numpy array
    X_train_electrostatic = np.vstack(train_features)
    X_train_electrostatic_df = pd.DataFrame(X_train_electrostatic)  
    X_train_electrostatic_df.reset_index(drop=True, inplace=True)
    print("X_train_electrostatic_df shape: ", X_train_electrostatic_df.shape)

    X_test_electrostatic_df['PDB_IDs'] = test_set_comps['PDB_IDs']
    X_train_electrostatic_df['PDB_IDs'] = train_set_comps['PDB_IDs']
    print("X_test_electrostatic_df shape: ", X_test_electrostatic_df.shape)

    print("X_train_electrostatic_df shape: ", X_train_electrostatic_df.shape)

    # Save protein IDs
    #test_protein_IDs = test_set_comps['PDB_IDs']
    #test_protein_IDs.reset_index(drop=True, inplace=True)
    #df = pd.DataFrame(test_protein_IDs)
    #df.to_csv('test_electrostatic_protein_IDs.csv')

    #train_protein_IDs = train_set_comps['PDB_IDs']
    #train_protein_IDs.reset_index(drop=True, inplace=True)
    #df = pd.DataFrame(train_protein_IDs)
    #df.to_csv('train_electrostatic_protein_IDs.csv')

    X_train_electrostatic_df.to_csv('X/X_train_electrostatic_p' + str(p) + '_L' + str(L) + '.csv')

    X_test_electrostatic_df.to_csv('X/X_test_electrostatic_p' + str(p) + '_L' + str(L) + '.csv')

# Need to cancel out ligand atoms charges
def get_protein(complex_file_path):
    print(complex_file_path)
    protein_file_path = complex_file_path[0:50] + "_protein_modified.pqr"
    print(protein_file_path)


    # Copy the original file to the new file location
    shutil.copyfile(complex_file_path, protein_file_path)
    # Read the original content from the new file
    with open(protein_file_path, 'r') as file:
        lines = file.readlines()

    # Modify the lines if the substring within indices 17:20 is 'SUB'
    modified_lines = []
    for line in lines:
        if line[17:20] == 'SUB':
            #print("Original line:", line.strip())
            # Modify the line
            modified_line = line[:54] + '  0.0000 ' + line[63:]
            #print("Modified line:", modified_line.strip())
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    # Write the modified lines back to the new file
    with open(protein_file_path, 'w') as file:
        file.writelines(modified_lines)

    return protein_file_path


def get_ligand(complex_file_path):
    print(complex_file_path)
    ligand_file_path = complex_file_path[0:50] + "_ligand_modified.pqr"
    print(ligand_file_path)


    # Copy the original file to the new file location
    shutil.copyfile(complex_file_path, ligand_file_path)
    # Read the original content from the new file
    with open(ligand_file_path, 'r') as file:
        lines = file.readlines()

    # Modify the lines if the substring within indices 17:20 is 'SUB'
    modified_lines = []
    for line in lines[:-2]:
        if line[17:20] != 'SUB':
            #print("Original line:", line.strip())
            # Modify the line
            modified_line = line[:54] + '  0.0000 ' + line[63:]
            #print("Modified line:", modified_line.strip())
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    # Write the modified lines back to the new file
    with open(ligand_file_path, 'w') as file:
        file.writelines(modified_lines)
        file.write("TER\n")
        file.write("END")


    return ligand_file_path

if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    
    # Call the function
    get_features(p, L)