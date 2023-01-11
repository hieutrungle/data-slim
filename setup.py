from setuptools import setup

setup(
    name="compression-vqvae",
    py_modules=["compression-vqvae"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "matplotlib", "netcdf4"],
)
