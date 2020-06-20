import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def requirements(fname):
    requires = []
    with open(fname) as file:
        for line in file:
            requires.append(line.strip())
    return list(requires)


setuptools.setup(
    name="physics",
    version="0.0.1",
    author="Rainer Weinberger",
    license="GNU General Public License v3 (GPLv3)",
    author_email="rainer.weinberger@cfa.harvard.edu",
    description="Useful physics stuff for python",
    install_requires=requirements("requirements.txt"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/rainerweinberger/physics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Operating System :: OS Independent",
    ],
)
