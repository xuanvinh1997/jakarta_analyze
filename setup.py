from setuptools import setup, find_packages

setup(
    name="jakarta_analyze",
    version="0.1.0",
    description="Tools for analyzing Jakarta traffic camera footage",
    author="Jakarta Smart City Traffic Safety Team",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'jakarta_analyze': ['config/*.yml', 'config/*.yaml'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'tensorflow',
        'torch',
        'matplotlib',
        'psycopg2-binary',
        'boto3',
        'pyyaml',
        'requests',
        'schedule',
    ],
    entry_points={
        'console_scripts': [
            'jakarta-analyze=jakarta_analyze.cli:main',
        ],
    },
    python_requires='>=3.6',
)