# The Thunderstorm Nowcasting Tool (ThunderCast)

The ThunderCast team is committed to the open-source science initiative. As a part of this, this code repository 
contains files used in the research presented in the following publication:

If you use any of these files in entirety or part, please cite our paper. 

## Python Environment

- environment.yml
    - The command `conda env create -f environment.yml` will create this environment if conda is installed.
- package_versions.txt
    - Contains the particular python package versions used in this research.

## Collecting and Formatting Data

1. MRMS_grib2netcdf.py
    - Function for converting gribs of radar data at -10 degrees celsius to netcdfs.
2. run_grib2nc.py 
    - Implement MRMS_grib2netcdf.py for datetimes of interest. 
3. save_max_radar.py
    - Save files with the maximum radar in 60 minutes (reduces time for collecting data patches to do this ahead of time).
4. data_patches.py
    - Gets data from the archive and formats it into 320x320 km patches. 
    - Use LATLONDT_patch.config to provide a list of dates to data_patches.py
5. make_model_file_list.py
    - Make txt file where each line is a file to be used in training/validation/testing.
6. determine_dataset_stats.py
    - Get the mean and standard deviation of all the model inputs for normalizing the dataset. 
7. nc_to_tensor.py
    - To avoid time bottlenecks during training/valiation/testing, convert the netcdfs to torch tensors and save in .pt files.
8. LATLONDT_patch.config
    - A configuration file containing latitudes, longitudes, and datetimes for collecting data patches. 
    
## Model Training/Validation/Testing

1. PL_Metrics.py
    - Custom metrics for model evaluation like Critical Success Index, etc. 
2. torchlightning_main.py
    -  Train and validate or test the deep learning model. The model configured is a U-Net convolutional neural network. Customizable with code modification.
3. run_model.sh
    - Run torchlightning_main.py using sbatch and slurm (train/validate).
4. run_test.sh
    - Run torchlightning_main.py using sbatch and slurm (testing).
5. extract_TB_data.py
    - During training/validation/testing Lightning saves the desired model statistics in a tensorboard file. This file gets the information out of the tensorboard file and plots it as needed. 
6. run_tbparse.sh
    - Run extract_TB_data.py in a cluster environment (slurm). 
6. sort_test_set.py
    - For additional testing, this file sorts the test dataset into sub testing sets (by region for example).

## Predictions

1. torchlightning_predict.py 
    - Create prediction plots (single) for specified times. 
2. plot_pred_subplots.py
    - Plot predictions in a subplot format with tiled images to show predictions through time. 

## Other Plotting Tools  
1. plot_dataset_info.py
    - Make bar plots to show number of files in each month and each climate region. 
2. plot_test_stats.py
    - Make performance and attribute diagrams using statistics generated when testing the model. 
3. plot_glm.py
    - Plot GLM flash extent density with satellite imagery in a paneled figure. 
4. climate_map.py
    - Create a map of the United States with the climate regions colored.
5. plot_rgb.py
    - Create a RGB image from ABI data. 

## Other Files
1. solar_angles.py
    - Contains code for calculating the satellite zenith and azimuth angles. 
2. cc_analysis.py
    - Contains code for collecting information about a point of interest in the Cape Canaveral, FL area (radar within 7km, prediction value, lat, lon, etc).
3. cc_stats.py
    - Determines statistics (mean, std) for the sea breeze dataset. 

** Note: The authors' formal training is not in Computer Science. We are always learning new "best practice" coding methods,
and there may be more efficient ways to set-up the python code presented here. 