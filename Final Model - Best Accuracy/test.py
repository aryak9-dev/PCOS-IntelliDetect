import kagglehub

# Download latest version
path = kagglehub.dataset_download("prasoonkottarathil/polycystic-ovary-syndrome-pcos")

print("Path to dataset files:", path)