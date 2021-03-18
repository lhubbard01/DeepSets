from setuptools import setup

setup(
      name="deepsets",
      version="0.0.1",
      author="Lyle Hubbard",
      author_email="lhubbard@pdx.edu",
      license="MIT",
      packages=["deepsets","deepsets.model","deepsets.utils", "deepsets.datasets"],
      

      install_requires=[
        "torch",
        "matplotlib",
        "torchvision",
        "tqdm",
        "pillow",
        "torchnet",
        "numpy",
        ],



        #scripts=["script/MnistSum/run_train"] #"scripts/MnistSum/eval"]
      )
