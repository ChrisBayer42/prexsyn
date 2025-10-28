from setuptools import find_packages, setup

setup(
    name="prexsyn-third_party",
    version="0.1",
    packages=find_packages(where="guacamol", include=["guacamol", "guacamol.*"]),
    package_dir={"guacamol": "guacamol/guacamol"},
)
