from setuptools import find_packages, setup

package_name = 'pointcloud_xyzrgb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='hyebingim5@gmail.com',
    description='GPU accelerated PointCloud2 generator from depth+RGB images',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'pointcloud_gpu_node = pointcloud_xyzrgb.point_cloud_xyzrgb:main',
        ],
    },
)
