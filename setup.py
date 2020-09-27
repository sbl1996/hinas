#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from setuptools import find_packages, setup

IMPORT_NAME = 'hinas'
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

about = {}
with open(HERE / IMPORT_NAME / '_version.py') as f:
    exec(f.read(), about)


def parse_requirements(fname='requirements.txt', with_version=True):
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4setup.py
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


setup(
    name='hinas',
    version=about['__version__'],
    description="HrvvI's NAS library",
    long_description=README,
    long_description_content_type='text/markdown',
    author='HrvvI',
    author_email='sbl1996@126.com',
    python_requires='>=3.6.0',
    url='https://github.com/sbl1996/hinas',
    packages=find_packages(exclude=('tests',)),
    install_requires=parse_requirements("requirements.txt"),
    license='MIT',
)

