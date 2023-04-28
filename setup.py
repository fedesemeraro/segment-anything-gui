from setuptools import find_packages, setup


setup(
    name="segment_anything_gui",
    version="0.0.1",
    author="Federico Semeraro and Alex Quintart",
    description="A matplotlib GUI to run the Segment Anything Model.",
    url="https://github.com/fsemerar/segment-anything-gui",
    project_urls={
        "Bug Tracker": "https://github.com/fsemerar/segment-anything-gui/issues",
    },
    platforms=["Linux", "Mac", "Windows"],
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "matplotlib", 
        "opencv-python", 
        "ipympl",
        "segment-anything"
    ],
)
