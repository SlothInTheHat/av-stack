from setuptools import setup

setup(
    name="avstack_control",
    version="0.1.0",
    packages=["avstack_control"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/avstack_control"]),
        ("share/avstack_control", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["controller_node = avstack_control.controller_node:main"]},
)
