AdaBoost
========

[![Build Status](https://travis-ci.com/codezonediitj/adaboost.svg?branch=master)](https://travis-ci.com/codezonediitj/adaboost) [![Join the chat at https://gitter.im/codezoned2017/Lobby](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/codezoned2017/Lobby) ![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

About Us
--------

We are some machine learning enthusiasts who aim to implement the adaboost algorithm from scratch.

Technologies
------------

We are using the following technologies in our project,

1. C++
2. Python
3. CUDA C
4. Google Test
5. Boost.Python

Building from source
--------------------
**Linux**

1. Clone Repository to local machine `git clone https://github.com/codezonediitj/adaboost`
2. Move to back to parent directory, `cd ../`
3. Execute, `mkdir build-adaboost`
4. Execute, `cd build-adaboost`
5. Execute, `cmake -D[OPTIONS] ../adaboost`
6. Execute, `make`. Do not execute, `make -j5` if you are using `-DINSTALL_GOOGLETEST=ON` otherwise `make` will try to link tests with `gtest gtest_main` before `GoogleTest` is installed into your system.
7. To test, run, `./bin/*`. Ensure that you have used the option `-DBUILD_TESTS=ON` in step 5 above.

**Windows**

1. git clone https://github.com/codezonediitj/adaboost
2. Move to back to parent directory, `cd ../`
3. Execute, `mkdir build-adaboost`
4. Execute, `cd build-adaboost`
5. Install CMake from - https://cmake.org/download/
- After downloading locate the bin folder under CMake directory and copy it's path(Like if the folder is located in C drive's Program files,path can be -C:\Program Files\cmake\bin).
- In the search bar on task bar type environment variables. Then click on the `Edit the System Variables`.
- The system properties dialog box will open, click on `Environment Variables`.
- Click on `Path` under system variables window. Then click `Edit` under the same.
- The edit enviornment dialog box will open, Click `New` and add the copied address of bin folder. Then Click `OK`.
- Similarly add the paths to your compilers (for example - if   it's MinGW add path to its bin folder for instance C:\MinGW\bin)
Also add path to the IDE used.
- To check that installation is done properly 
run: `cmake --version` on git bash.
It should give you the version of cmake as the output.

6. Open cmake GUI and put the adaboost directory as source code in the source code field and build-adaboost directory in the build binaries field.

7. By default `BUILD_TESTS`,`BUILD_CUDA` are turned off, you can check these options if you want to perform tests and want CUDA support , then click configure and generate to build the files .

8. Before checking the `BUILD_TESTS` field , install google test from :- https://github.com/google/googletest and build it using CMake.

9. For CUDA support you need to have a CUDA compiler and you should add it's path to environment variables therefore before checking `BUILD_TESTS` field  you should have a CUDA compiler.


We provide the following options for `cmake`,

1. `BUILD_TESTS`

By default `OFF`, set it to `ON` if you wish to run the tests. Tests are stored in the `bin` under your build directory.

2. `INSTALL_GOOGLETEST`

By default `ON`, set it to `OFF` if you do not want to update the already existing GoogleTest on your system. Note that it uses [this release](https://github.com/google/googletest/archive/release-1.10.0.tar.gz) of googletest.

3. `CMAKE_INSTALL_PREFIX`

Required for installing if not installing to `/usr/local/include` on Linux based systems. Defines the path where the library is to be installed.


Installing
----------

Follow the steps for building from source. After that run the following,

**Linux**
```
sudo make install
```
**Windows**
```
cmake install <path to your build binaries directory>
```

How to contribute?
------------------

Follow the steps given below,

1. Fork, https://github.com/codezonediitj/adaboost
2. Execute, `git clone https://github.com/codezonediitj/adaboost/`
3. Change your working directory to `../adaboost`.
4. Execute, `git remote add origin_user https://github.com/<your-github-username>/adaboost/`
5. Execute, `git checkout -b <your-new-branch-for-working>`.
6. Make changes to the code.
7. Add your name and email to the AUTHORS, if you wish to.
8. Execute, `git add .`.
9. Execute, `git commit -m "your-commit-message"`.
10. Execute, `git push origin_user <your-current-branch>`.
11. Make a PR.

That's it, 10 easy steps for your first contribution. For future contributions just follow steps 5 to 10. Make sure that before starting work, always checkout to master and pull the recent changes using the remote `origin` and then start following steps 5 to 10.

See you soon with your first PR.

Guidelines
----------

We recommend you to introduce yourself on our [gitter channel](https://gitter.im/codezoned2017/Lobby). You can include the literature you have studied relevant to adaboost, some projects, prior experience with the technologies mentioned above, in your introduction.

Please follow the rules and guidelines given below,

1. For Python we follow the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).
2. For C++ we follow our own coding style mentioned [here](https://github.com/codezonediitj/adaboost/issues/3#issuecomment-581055358).
3. For C++ documentation we follow, Doxygen style guide. Refer to various modules in the existing `master` branch for the pattern.
4. Follow the Pull Request policy given [here](https://github.com/codezonediitj/adaboost/wiki/Pull-Request-Policy). All changes are made through Pull Requests, no direct commits to the master branch.

Keep contributing!!
