from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DRL-ORAN-EnergySaving",
    version="0.1.0",
    author="bolun",
    author_email="bolun_zhangzbl@outlook.com",
    description="A Deep Reinforcement Learning approach for energy saving in O-RAN.",
    url="https://github.com/your-username/DRL-ORAN-EnergySaving",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)