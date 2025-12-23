from setuptools import setup
import os
from glob import glob

package_name = 'dwa_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일 설치 ---
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        # param yaml 설치---
        (os.path.join('share', package_name, 'param'),
         glob('param/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
    ],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='hyebingim5@gmail.com',
    description='DWA local planner node using CUDA distance map',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run dwa_nav dwa_node
            'dwa_node = dwa_nav.dwa_node:main',
        ],
    },
)

