from setuptools import setup

package_name = 'gps_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # 패키지 인덱스 등록
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml 설치
        ('share/' + package_name, ['package.xml']),

        # launch 파일까지 만들 거면 여기에 추가 (없으면 이 줄은 빼도 됨)
        # ('share/' + package_name + '/launch', ['launch/gps_nav.launch.py']),
    ],
    install_requires=[
        'setuptools',
        # 이 파일에서 실제로 쓰는 외부 파이썬 라이브러리들
        'pyubx2',
        'pynmeagps',
        'redis',
        'termcolor',
        'python-dotenv',
        'pyserial',
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='GPS + Unitree Go2 navigation server with RTK and path tracking',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run gps_nav planner_server
            #  → gps_nav/planner_server11.py 의 main() 실행
            'planner_server = gps_nav.planner_server11:main',
            # 나중에 DWA 노드도 패키지에 넣으면 여기 추가:
            # 'dwa_node = gps_nav.dwa_command_node:main',
        ],
    },
)

