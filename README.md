# Medical AI for [COVID-19](https://www.google.com/covid19/)

Supervised calsification patients in covid19, Community acquired pneumonia (CAP) and other non-pneumonia by CT dicom files exams.


## Training flow

1. dataset.py : preprocessing dicom files dataset and save into .npy files
2. data_generator.py : customize data generator 
3. train_mode.py : CNN model in keras and tensorflow backend

### 1. dataset.py

This modoule convert all dicom files as a patient into 3D numpy array and saved into dataset as 
patient id.npy.

Dicom files per a patient in a folder in dataset_dcm directory and folder name is patient_id
first dicom file use as refernce for finding pixel dimension and spacing :
```
	RefDs = dicom.read_file(lstFilesDCM[0])    
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
```

and then show plot last dicom file in 3 2D (x,y), (z,y) and (z,x) and 3d plot 
for to show accuracy npy files in end of preprocessing.

### 2. data_generator.py

Load .npy files in dataset_npy directory as 3D numpy array:
```
	X[i,] = np.load('dataset_npy\\' + ID + '.npy')
```

Dimension is 2D and dicom files count is as channel count:
```
	dim=(512, 512), n_channels=488,
```

### 3. train_model.py

Load train and validation IDs and labels as Dict:
```
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
```

See also in this [article](https://pubs.rsna.org/doi/10.1148/radiol.2020200905).
