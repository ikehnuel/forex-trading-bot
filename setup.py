from setuptools import setup, find_packages

setup(
    name='forex-trading-bot',
    version='0.1.0',
    author='Emmanuel Ikechukwu',
    author_email='ikehnuel@gmail.com',
    description='A Forex trading bot that implements various trading strategies and indicators.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
        'numpy',
        'pandas',
        'requests',
        'matplotlib',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)