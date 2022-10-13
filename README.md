# DECODE

To run the code you need to run the `main` function.
`main` gets 5 parameters:
```
output_folder: str,
ref_folder: str,
mixes_folder: str,
true_prop_folder: str,
output_folder_final: str,
index_dataset: int = 0
```
`output_folder` - in this folder the model saves its temporary models, after learning with simulated data  (can be tmp folder) </br>
`output_folder_final` - In this folder the model saves the final results </br>
`mixes_folder` - this folder contains the datasets (bulk expressions) </br>
`true_prop_folder` - this folder contains the desired cells of the datasets (for each dataset 'x.tsv' there is a file 'TruePropx.tsv' with one column - the desired cells). we are going to predict these cells</br>
`ref_folder` - this folder contains signature matrices </br>
`index_dataset` - The index of the dataset in the `mixes_folder` (the default is to run the DECODE algorithm on the first file) 
