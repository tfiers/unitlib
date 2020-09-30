from setuptools import find_packages, setup


GITHUB_URL = "https://github.com/tfiers/yunit"

with open("ReadMe.md", mode="r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="yunit",
    description="📐 Physical units for NumPy arrays ⏱ Fast • Simple • High voltage",
    author="Tomas Fiers",
    author_email="tomas.fiers@gmail.com",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    project_urls={"Source Code": GITHUB_URL},
    classifiers=(
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    packages=find_packages("src"),
    package_dir={"": "src"},  # This means: "Root package can be found in src dir"
    python_requires=">= 3.6",
    #   `typing-extensions` (see below) does not provide a backport for
    #   `typing.Protocol` (which we use in `type_aliases.py`) for Python 3.5 . Hence
    #   this minimum Python version.
    install_requires=(
        "numpy",
        "typing-extensions; python_version < '3.8'",
        #   This provides backports of `typing.Literal` etc, which are not in the
        #   standard library until Python 3.8.
        "dataclasses; python_version < '3.7'",
    ),
    # Get package version from git tags
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "dirty-tag",
    },
)
