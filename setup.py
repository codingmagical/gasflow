# Copyright 2022 Zeeland(https://github.com/Undertone0809/). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import pathlib

# here = pathlib.Path(__file__).parent.resolve()
# long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="gasflow",
    version="0.0.5",
    author="qxd",
    author_email="qxd_cup@163.com",
    description="A python library for calculating natural gas properties and gas pipeline simulation",
    package_data={'your_package': ['/Users/qxd/PycharmProjects/Gaspy/gasflow/data/*.npy']},
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/codingmagical/gaspy",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'sympy', 'scipy', 'tqdm'],
    python_requires='>=3.9',
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

    ],
    keywords="gas, property, simulation",
)
