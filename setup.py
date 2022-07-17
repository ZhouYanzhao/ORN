from setuptools import setup, find_packages


setup(
    name='orn',
    version='1.0.0',
    description='Oriented Response Networks (Zhou, CVPR2017)',
    url='https://github.com/ZhouYanzhao/ORN',
    author='Zhou, Yanzhao',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True
)
