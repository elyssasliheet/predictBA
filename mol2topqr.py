import os
import pandas as pd
import time

test_set_comps = pd.read_csv("test_set_comps_final_for_CNN.txt", names = ['PDB_IDs'])
train_set_comps = pd.read_csv("train_set_comps_final_for_CNN.txt", names = ['PDB_IDs'])
all_comps = pd.concat([test_set_comps, train_set_comps], axis = 0)
# Now convert file types
fail = 0
for comp in all_comps['PDB_IDs']:
  start = time.time()
  print("Running comp: ", comp, "+++++++++++++++++++++++++++++++++++++++++++++++++")
  dir = "/users/esliheet/esliheet/TDL-BP/v2007/" + comp 
  print(dir)
  os.chdir(dir) # change to directory
  #forcefield="AMBER"
  input_filename = comp + "_ligand.mol2"
  output_filename = comp + "_ligand.pqr"
  os.popen(f'obabel {input_filename} -opqr -O {output_filename}')
  if os.path.isfile(output_filename):
    print(f"{output_filename} was successfully created.")
  else:
    print(f"Failed to create {output_filename}.")
    fail = fail + 1
  end = time.time()
  total_time = end-start

  #print("total time for " ,comp, " : " , total_time , "seconds")
  with open(output_filename, 'r') as f:
    lines = f.readlines()
    print("OPENED FILE")
    # Process each line
    new_lines = []
    for line in lines:
     # Split the line by whitespace
     parts = line.split()
     if parts[0] == 'ATOM':
      #print("len", len(parts))
      #print(parts)
      # Remove the column of all 'A's and the last column
      new_line_parts = parts[:4] + parts[5:-1]
      # Join the modified parts back into a line
      new_line = ' '.join(new_line_parts) + '\n'
      # Append the modified line to the new lines list
      new_line = "{:>4s}{:>7s}{:>5s}{:>4s}{:>6s}{:>12s}{:>8s}{:>8s}{:>8s}{:>6s}\n".format(
        parts[0], parts[1], parts[2], parts[3], parts[5], parts[6], parts[7], parts[8], parts[9][0:7], parts[10])
      new_lines.append(new_line)
      # Write the modified lines back to a new file
  with open(comp + "_ligand_formatted.pqr", 'w') as f:
      f.writelines(new_lines)

print("FAILS", fail)
