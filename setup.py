from setuptools import setup, find_packages

setup(
    name="spdn",
    version="0.5.0",
    author="Calvin Leighton",
    description="Speckle Pattern Denoising Network for OCT imaging",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
)
