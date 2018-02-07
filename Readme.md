# Barracuda Concurrency Bug Suite

This repository contains the concurrency bug suite used to evaluate Barracuda (See our PLDI '17 paper for details: Eizenberg, Ariel, Yuanfeng Peng, Toma Pigli, William Mansky, and Joseph Devietti. "BARRACUDA: Binary-level Analysis of Runtime RAces in CUDA programs." In Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation, pp. 126-140. ACM, 2017.)

# Prerequisites 

We've tested the code with both CUDA 7.5 and 8.0.  Most of the tests can be compiled with SM >= 3.5, but some tests (e.g. those involving dynamic parallelism) needs SM >= 5.0.   


# Run Tests
Thanks to Toma Pigli, we have some python scripts that can be used to compile & run the tests.  To run the tests, simply execute the following:

`python runTests.py`

# Build Tests manually

If you want to compile the tests manually, we've provided a Makefile for each set of the tests.  

For example, if you want to compile all 'racey' tests, simply do the following: 

`cd racey`
`make -j16`

We can also modify the Makefile if you need to change any flag for the compilation. 

# Contact us

If you have any questions regarding these tests or Barracuda, feel free to drop Yuanfeng Peng an email at yuanfeng@cis.upenn.edu.

 
