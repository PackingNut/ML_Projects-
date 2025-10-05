import kagglehub



# Download and import the latest version of the datatset we are using
path = kagglehub.dataset_download("jayanthbottu/labeled-deepfake-image-collection")

print("Path to dataset files:", path)