from setuptools import setup

package_name = "swarm_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/supervisor_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Meghlad",
    maintainer_email="meghlad06@gmail.com",
    description="Bringup + rclpy bridge nodes for the cooperative swarm ROS 2 graph.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "estimate_publisher = swarm_bringup.estimate_publisher:main",
            "plan_publisher = swarm_bringup.plan_publisher:main",
        ],
    },
)
