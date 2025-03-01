###############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
###############################################################################
"""
Project setup with setuptools
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
import pathlib
import re

# this will come in handy, probably
cwd = pathlib.Path(__file__).parent.resolve()

# Parse long description from README.md file
with open("README.md") as f:
    lines, capture = [], False
    for line in f:
        s = line.strip()
        if re.match(r"#\s*[Aa]bout", s):
            capture = True
        elif re.match("^#", s):
            break
        elif capture is True:
            lines.append(s)
    if lines:
        long_description = " ".join(lines)
    else:
        long_description = "DISPATCHES project"


def read_requirements(input_file):
    """Build list of requirements from a requirements.txt file
    """
    req = []
    for line in input_file:
        s = line.strip()
        c = s.find("#")  # look for comment
        if c != 0:  # no comment (-1) or comment after start (> 0)
            if c > 0:  # strip trailing comment
                s = s[:c]
            req.append(s)
    return req


class SpecialDependencies:
    """
    The following packages require special treatment, as they change rapidly between release cycles.
    Two separate lists of dependencies are kept:
    - for_release: to be used when cutting a release of DISPATCHES
    - for_prerelease: to be used for the prerelease version of DISPATCHES (i.e. the `main` branch, and all PRs targeting it)
    """
    # idaes-pse: for IDAES DMF -dang 12/2020
    for_release = [
        "idaes-pse=>1.11.0"
    ]
    for_prerelease = [
        "idaes-pse>=1.11.0"
    ]


SPECIAL_DEPENDENCIES = SpecialDependencies.for_prerelease


########################################################################################

setup(
    name="dispatches",
    url="https://github.com/gmlc-dispatches/dispatches",
    version="0.2.0dev0",
    description="GMLC DISPATCHES software tools",
    long_description=long_description,
    long_description_content_type="text/plain",
    author="DISPATCHES team",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="market simulation, chemical engineering, process modeling, hybrid power systems",
    packages=find_namespace_packages(),
    python_requires=">=3.6, <4",
    install_requires=[
        "pytest",
        # we use jupyter notebooks
        "jupyter",
        # for visualizing DMF provenance
        "graphviz",
        "gridx-prescient",
        "nrel-pysam",
        *SPECIAL_DEPENDENCIES
    ],
    package_data={"": ["*.json"]},  # Optional
)
