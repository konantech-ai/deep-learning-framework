""" Setup script for konanai package """
"""
Caution! If you want to reinstall after changing this file,
you must delete the build and PACKAGENAME-egg.info directories before reinstalling.
"""

import setuptools

package_name = 'konanai'
package_src_dir = 'konanai'
package_dst_dir = 'konanai'     # This variable is also used as a keyword
library_src_dir = '../../out/Release'
library_dst_dir = 'konanai.'    # This variable is also used as a keyword

setuptools.setup(
    name = package_name,
    version = '0.1',
    author = 'Konan Technology Inc.',
    author_email = 'dhyoon@konantech.com',
    description = 'Konan Technology AI engine, datasets, tutorials, and etc.',
    url='http://10.10.18.132:6118/vision-ai-lab/kai2021',
    python_requires = '>=3.9.5',
    install_requires = open('requirements.txt').read().splitlines(),
    include_package_data=True,
    packages=setuptools.find_packages(where='.') + [library_dst_dir],
    package_dir={
        package_dst_dir: package_src_dir,
        library_dst_dir: library_src_dir},
    package_data={
        library_dst_dir: ['*.dll', 'libkonanai_python.so']
    },
    zip_safe=False
)
