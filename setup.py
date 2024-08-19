# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['ideas']

package_data = \
{'': ['*']}

install_requires = \
['beartype==0.15.0',
 'configobj>=5.0.8',
 'figrid>=0.1.6',
 'jinja2>=3.1.2',
 'jsonschema>=4.17.3',
 'numpy==1.26.4',
 'opencv-python==4.10.0.84',
 'pyarrow>=13.0.0',
 'tabulate>=0.9.0']

extras_require = \
{':extra == "plotting"': ['bokeh>=3.1.0'],
 'dev': ['ipykernel>=6.20.1',
         'debugpy==1.6',
         'poetry2setup>=1.1.0',
         'coverage-badge>=1.1.0'],
 'extras': ['pandas>=1.5.3', 'scikit-image>=0.24.0,<0.25.0', 'scipy>=1.10.1'],
 'ideas-commons': ['ideas-commons @ '
                   'git+https://github.com/inscopix/ideas-commons@1.19.0'],
 'ideas-schemas': ['ideas_schemas @ '
                   'git+https://github.com/inscopix/ideas-schemas@python-wrapper'],
 'plots': ['matplotlib'],
 'test': ['pytest>=7.2.0', 'interrogate>=1.5.0', 'coverage>=7.2.1'],
 'test-isx': ['ideas-data @ git+https://github.com/inscopix/ideas-data@2.0.0',
              'isx @ git+https://github.com/inscopix/py_isx@main']}

setup_kwargs = {
    'name': 'ideas-python-utils',
    'version': '23.11.21',
    'description': 'Python utilities for tools in IDEAS',
    'long_description': '![coverage](coverage.svg)\n\n# IDEAS Python Utils\n\nThis repository contains utilities and functions to help you\nbuild tools for IDEAS. \n\n\n## Installation\n\nYou do not have to install this to use (unless you are developing this).\nIf so, see [below](#developing)\n\n\n## Usage\n\n\n### In a project using [Pipfile](https://github.com/pypa/pipfile)\n\nIf you\'re building a tool in IDEAS, you are probably working \non a project that uses a `Pipfile`. Include this repo in your IDEAS tool\nby adding this to your  `Pipfile`:\n\n```bash\n[packages]\nideas-python-utils = {ref = "main", git = "https://${IDEAS_GITHUB_TOKEN}@github.com/inscopix/ideas-python-utils.git"}\n```\n\n\n> You can omit the `IDEAS_GITHUB_TOKEN` when this repo becomes public. \n\n\n\n\n## Developing \n\n\n### Prerequisites\n\n- python\n- make\n- git\n- You should have SSH keys set up with github\n\nIf you are working on developing this, download and install using:\n\n\n```bash\ngit clone git@github.com:inscopix/ideas-python-utils.git\ncd ideas-python-utils\npoetry install  # this is a "editable" install\n```\n\nIf you use Jupyter Lab, and you want this kernel available to your\nglobal install of Jupyter Lab, use this:\n\n```bash\nmake jupyter\n```\n\nThis will create a kernel called `ideas_python_utils` that you can access\nfrom Jupyter (Lab or notebook). \n\n## Run tests \n\n### Locally\n\n```bash\nmake test\n```\n\n\n### On remote. \n\nTests should run via Github Actions on push/merge to `main`. This is automatic. \n\n\n## License ',
    'author': 'Inscopix, Inc.',
    'author_email': 'support@inscopix.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.9,<3.12',
}


setup(**setup_kwargs)

