:tocdepth: 1

Installation
============

*scarlet* has several dependencies that must be installed prior to installation:

#. numpy_
#. scipy_
#. proxmin_ (proximal algorithms to optimize the likelihood)
#. pybind11_ (integrate C++ code into python)
#. peigen_ (used to load the Eigen_ headers if they are not already installed)
#. autograd_ (needed to calculate gradients during optimization)

Optional Dependencies, but probably useful for most users:

#. matplotlib_
#. astropy_

The easiest way is using a combination of `conda` and `pip` installers:

::

    conda install numpy scipy astropy pybind11
    pip install proxmin peigen autograd

If you don't work with `conda`, `pip` alone will do as well.
Then go to a directory that should hold the scarlet code and get the scarlet repository
from github, and build and install it:

::

    git clone https://github.com/pmelchior/scarlet.git
    cd scarlet
    python setup.py install


*scarlet* requires the Eigen_ library headers, which are downloaded automatically when using the
command above.
If you already have a local version of Eigen_ and don't want to download the headers, use

::

    python setup.py build_ext -I<full path to your Eigen header files>
    python setup.py install

.. warning::
    `build_ext` does not accept relative paths, so `<full path to your Eigen header files>`
    must be a full path.


For OS X Users
--------------

For some reason the default XCode build system attempts to use the deprecated stdlc++ library instead of the
newer libc++. So you may receive the error
`clang: warning: libstdc++ is deprecated; move to libc++ with a minimum deployment target of OS X 10.9 [-Wdeprecated]
ld: library not found for -lstdc++`.
If this is the case then before you run the setup script you will need to run:
::

    xcode-select --install
    sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /

Building the Docs (*scarlet* developers only)
---------------------------------------------

You need to install several extra packages:

#. sphinx_ (required to build the docs)
#. nbsphinx_ (required to compile notebooks)
#. numpydoc_ (allow for numpy style docstrings)


Then navigate to the `docs` directory and type
::

    make html

and a local copy of the current docs will be available in the `docs/_build/html` folder.
The home page is available at `docs/_build/html/index.html`.

.. _numpy: http://www.numpy.org
.. _proxmin: https://github.com/pmelchior/proxmin/
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _peigen: https://github.com/fred3m/peigen
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _autograd: https://github.com/HIPS/autograd
.. _matplotlib: https://matplotlib.org
.. _astropy: http://www.astropy.org
.. _sphinx: http://www.sphinx-doc.org/en/master/
.. _nbsphinx: https://nbsphinx.readthedocs.io/en/0.4.2/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
.. _scipy: https://www.scipy.org/
