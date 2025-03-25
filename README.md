# Medusa: Scalable Multi-View Biometric Sensing in the Wild with Distributed MIMO Radars

We present our artifacts for Medusa, a MIMO multi-view sensing system for non-contact vital sign monitoring. 
This repo includes complete hardware designs—PCB layouts created in Altium Designer and antenna designs developed with HFSS and ADS, ready for manufacturing—as well as software for MIMO signal processing and an unsupervised learning model. Additionally, we provide a point cloud demo based on our \(16 \times 16\) UWB MIMO array system. For users interested in training their own model, we supply training data; for those evaluating performance, we include comprehensive experimental datasets.


- **UWB_MIMO_Platform:**  
  The folder "Hardware Design" contains PCB design files for the motherboard and radar element board created in Altium Designer, along with the corresponding Gerber manufacturing files. The "Antenna" folder includes the Vivaldi antenna design used in \system. The "MIMO Array Processing" folder comprises MATLAB code for AoA and radar cube processing, as well as example data.

- **UWB_MIMO_Platform - Hardware Design:**  
  The "Radar Element" folder contains the PCB design for the radar element small board, the "X4_test_manufacturing files" folder includes the corresponding Gerber manufacturing files, and the "Antenna" folder provides the Vivaldi antenna design.

- **UWB_MIMO_Platform - MIMO Array Processing:**  
  Due to upload file size limitations, we have included only two scripts for MIMO processing. "radardata_1x16.mat" is provided as a sample dataset for AoA processing. If you prefer to process the data using "raw_radar_data", please note that it will require a longer processing time.

- **Unsupervised Model:**  
  "Model_Medusa" is the older version of the \system model, developed in TensorFlow and used for three years. "New_Model" contains the new vital sign decomposition model implemented in PyTorch, and "Training_Data" provides the sample dataset for training the nonlinear ICA contrastive learning model as described in the paper. "Performance.mat" and "features.m" are samples of the experimental data. If users do not wish to train the model, they can run the script to view the results.

- **UWB_MIMO_PointCloud:**  
  A demo of point cloud data using the \system UWB MIMO radar system.

