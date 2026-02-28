from setuptools import setup, find_packages


setup(
    name='savnce',
    version='0.1.0',
    packages=find_packages(
        where=".",  
        include=[
            "savnce*",
            "habitat*",
            "savnce_baselines*",
        ],
    ),
    install_requires=[
        'torch',
        'gym',
        'numpy>=1.16.1',
        'yacs>=0.1.5',
        'numpy-quaternion>=2019.3.18.14.33.20',
        'attrs>=19.1.0',
        'opencv-python>=4.10.0.84',
        'imageio>=2.2.0',
        'imageio-ffmpeg>=0.2.0',
        'scipy>=1.0.0',
        'tqdm>=4.0.0',
        'Pillow',
        'matplotlib',
        'librosa',
        'torchsummary',
        'tqdm',
        'moviepy',
        'tensorflow',
        'scikit-image'
    ],
    python_requires=">=3.9",
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
