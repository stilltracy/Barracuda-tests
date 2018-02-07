#!/usr/bin/python
from argparse import ArgumentParser
from collections import defaultdict
from cudaTestProgram import *
from racey import *
from spatial_offset_r import *
from spatial_offset_w import *
from temporal import *

testPrograms = raceyTests + spatialOffsetReadTests + spatialOffsetWriteTests + temporalTests


if __name__ == "__main__":

    # parse args, if any
    parser = ArgumentParser(description="This is a script to test cuda-memcheck's memory safety coverage")
    parser.add_argument('-m','--mode',choices=['memcheck','racecheck','all'],default='all',required=False)
    args = parser.parse_args()

    if args.mode == 'memcheck':
        testPrograms = spatialOffsetReadTests + spatialOffsetWriteTests + temporalTests
    elif args.mode == 'racecheck':
        testPrograms = raceyTests

    # setup output dirs
    CudaTestProgram.setupDirs()

    # run tests
    results = defaultdict(lambda:CudaTestResult())

    count = len(testPrograms)
    i = 0
    for t in testPrograms:
        i += 1
        if(int(20 * (i - 1) / count) != int(20 * i / count)):
            print "%i/%i\n" % (i, count);
        t.compileProgram()
        t.runTestCases()

        results[t.testMode].update(t.testResults)

    # print results
    for mode, testResult in results.iteritems():
        print '\n', mode.capitalize(),
        testResult.display()
