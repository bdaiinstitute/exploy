"""Setup file for the exporter-isaaclab package."""

from setuptools import setup, find_namespace_packages

setup(
    name="exporter-isaaclab",
    version="0.1.0",
    description="IsaacLab-specific ONNX environment exporter implementation",
    author="Robotics and AI Institute LLC dba RAI Institute",
    packages=find_namespace_packages(where="../../..", include=["exporter.frameworks.isaaclab*"]),
    package_dir={"": "../../.."},
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "numpy",
        "gymnasium==1.2.0",
        "tqdm",
        "isaacsim[all,extscache]==5.1.0",
    ],
)
