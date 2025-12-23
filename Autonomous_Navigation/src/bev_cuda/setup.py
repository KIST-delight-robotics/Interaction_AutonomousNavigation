import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bev_cuda'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament가 패키지를 인식하게 하는 부분
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml 설치
        ('share/' + package_name, ['package.xml']),
        # launch/ 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # param/ 설치
        (os.path.join('share', package_name, 'param'), glob('param/*.yaml')),
    ],
    install_requires=['setuptools', 'pycuda', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='t25307@kist.re.kr',
    description='BEV Image Generator using CUDA from PointCloud2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_node = bev_cuda.bev_node:main'
        ],
    },
)

