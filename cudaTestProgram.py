from collections import Counter, defaultdict
from math import ceil
import os
from subprocess import call, check_output, CalledProcessError, STDOUT
from shutil import rmtree
from time import time

import fnmatch, os, stat



class CudaError(object): # hacky enum, but more portable

    ARG = "invalid argument"
    GLB_R = "Invalid __global__ read"
    GLB_RAW = "RAW hazard detected at __global__"
    GLB_W = "Invalid __global__ write"
    GLB_WAR = "WAR hazard detected at __global__"
    GLB_WAW = "WAW hazard detected at __global__"
    LCL_R = "Invalid __local__ read"
    LCL_W = "Invalid __local__ write"
    SHR_R = "Invalid __shared__ read"
    SHR_RAW = "RAW hazard detected at __shared__"
    SHR_W = "Invalid __shared__ write"
    SHR_WAR = "WAR hazard detected at __shared__"
    SHR_WAW = "WAW hazard detected at __shared__"
    NO_ERR = "ERROR SUMMARY: 0 errors"
    NO_RACE = "RACECHECK SUMMARY: 0 hazards"
    OOM = "out of memory"
    PITCH_ARG = "invalid pitch argument"



class CudaWarning(object):
    SUB_OOR = 'subscript out of range'
    VAR_UNUSED = 'variable "*" was set but never used'
    VAR_USED_BEFORE_SET = 'variable "*" is used before its value is set'

    @staticmethod
    def subtractWarningSets(source, subtrahend):
        result = source.difference(subtrahend)
        for w in subtrahend:
            if '*' in w: # if contains a wildcard, apply wildcard
                result.difference_update(fnmatch.filter(result, w))
        return result



class CudaTestMode(object):

    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"



class CudaTestResult(object):

    COMPILER_ERROR = "compiler error" # don't print these in results
    COMPILER_NO_WARNINGS = "comiler no warnings"
    COMPILER_REPORTED = "compiler reported"
    COMPILER_UNEXPECTED = "compiler unexpected"
    TOOL_MISREPORTED = "tool misreported"
    TOOL_REPORTED   = "tool reported"
    TOOL_UNEXPECTED = "tool unexpected"
    TOOL_UNREPORTED = "tool unreported"

    ALL_COMPILER = [
            COMPILER_REPORTED,
            COMPILER_NO_WARNINGS,
            COMPILER_UNEXPECTED,
        ]

    ALL_TOOL = [
            TOOL_REPORTED,
            TOOL_UNREPORTED,
            TOOL_MISREPORTED,
            TOOL_UNEXPECTED,
        ]

    COMPILER = 'nvcc'
    COMPILER_VERSION = None


    @staticmethod
    def getCompilerVersion():
        if CudaTestResult.COMPILER_VERSION is None:
            cmd = [CudaTestResult.COMPILER, '-V']
            try:
                out = check_output(cmd, stderr=STDOUT)
                CudaTestResult.COMPILER_VERSION = out[out.lower().index('release'):].split()[-1]
            except CalledProcessError as exc:
                print 'ERROR: Could not determine {0} version'.format(CudaTestResult.COMPILER)
                print exc.output
                CudaTestResult.COMPILER_VERSION = 'V?.?.??'

        return CudaTestResult.COMPILER_VERSION


    def __init__(self, targetArchitecture=''):
        self.results = defaultdict(Counter)
        self.skipped = self.total = 0
        self.arch = targetArchitecture


    def add(self, toolResult):
        self.results[self.compileResult][toolResult] += 1
        self.total += 1


    def addSkipped(self, skipped):
        self.skipped += skipped
        self.total += skipped


    def display(self):
        print 'Completed {0} of {1} tests ({2} skipped)'.format(
                self.total - self.skipped, self.total, self.skipped
            )
        colTitles = ['{0:>11}'.format(' '.join(c.split()[1:])) for c in CudaTestResult.ALL_COMPILER]
        titleWidth = sum([len(ct) for ct in colTitles]) + len(colTitles)

        archInfo = ',arch={0}'.format(self.arch) if self.arch else ''
        compilerTitle = 'COMPILER: {0} ({1}{2})'.format(
                CudaTestResult.COMPILER, CudaTestResult.getCompilerVersion(), archInfo
            ).center(titleWidth)
        print '{0}|{1}'.format(' '*14,compilerTitle)
        print '{0}| {1}'.format(' '*14,' '.join(colTitles))

        print '{0}'.format('_'*(titleWidth+15))
        for i,t in enumerate(CudaTestResult.ALL_TOOL):
            cols = ['{0:11}'.format(self.results[c][t]) for c in CudaTestResult.ALL_COMPILER]
            print 'TOOL'[i] + ' {0:12}| {1}'.format(t.split()[1],' '.join(cols))


    def setDefaultCompileResult(self, compileResult):
        self.compileResult = compileResult


    def update(self, otherCudaTestResult):
        for k,v in otherCudaTestResult.results.iteritems():
            self.results[k].update(v)
        self.skipped += otherCudaTestResult.skipped
        self.total += otherCudaTestResult.total

        if self.arch is None and otherCudaTestResult.arch:
            # update arch if we have none
            self.arch = otherCudaTestResult.arch
        elif self.arch is not otherCudaTestResult.arch:
            # ignore arch if we're combining results from multiple architectures
            self.arch = None



class CudaTestCase(object):

    def __init__(self, expectedText, *args):
        self.expectedText = expectedText
        self.args = args


    def getArgsList(self):
        return list(self.args)


    def getExpectedText(self):
        return self.expectedText



class CudaTestProgram(object):

    BIN_DIR = 'bin'
    LOG_DIR = 'log'


    @staticmethod
    def setupDirs():
        # wipe executables on each run
        if os.path.exists(CudaTestProgram.BIN_DIR):
            rmtree(CudaTestProgram.BIN_DIR, ignore_errors=True)
        os.makedirs(CudaTestProgram.BIN_DIR)

        if not os.path.exists(CudaTestProgram.LOG_DIR):
            os.makedirs(CudaTestProgram.LOG_DIR)


    def __init__(self, 
            sourceFile, 
            testCases, 
            compiler='nvcc', 
            profileTool='cuda-memcheck', 
            testMode=CudaTestMode.MEMCHECK, 
            macros={},
            expectedCompilerWarnings=[],
            ):
        self.sourceFile = sourceFile
        self.profileTool = profileTool
        self.testCases = testCases
        self.testMode = testMode
        self.macros = macros

        self.compiler = compiler
        self.expectedCompilerWarnings = expectedCompilerWarnings

        # strip extension
        self.program = self.sourceFile[:self.sourceFile.rfind('.')]
        if self.program is self.sourceFile:
            self.program += '_out'
        # replace path separator if any
        self.program = self.program.replace(os.sep, '_')
        
        self.testResults = CudaTestResult(self.getTargetArchitecture())


    def compileProgram(self):
        cmd = [self.compiler, self.sourceFile]

        blankDefs = [k for k,v in self.macros.items() if v is None]
        valueDefs = [k + '=' + v for k,v in self.macros.items() if v is not None]
        defs = blankDefs + valueDefs # separate for print order
        if defs:
            cmd += ['-D' + d for d in defs]
            self.program += '_' + '_'.join(sorted(blankDefs) + sorted(valueDefs))

        self.qualifiedProgram = os.path.join(CudaTestProgram.BIN_DIR, self.program)
        cmd += ['-arch', self.getTargetArchitecture(), '-o', self.qualifiedProgram, '-rdc=true']

        try:
            #print "Running %s\n" % cmd
            compilerOut = check_output(cmd, stderr=STDOUT)
            self.parseCompilerOutput(compilerOut, self.expectedCompilerWarnings)
        except CalledProcessError as exc:
            self.testResults.setDefaultCompileResult(CudaTestResult.COMPILER_ERROR)
            print 'ERROR: Compilation failed: ', self.sourceFile
            print exc.output
            return
        
        status = os.stat(self.qualifiedProgram)
        os.chmod(self.qualifiedProgram, status.st_mode | stat.S_IEXEC)


    def getIgnoredWarnings(self):
        # this is used to filter out certain warnings in subclasses, if necessary
        return []


    def getTargetArchitecture(self):
        return 'sm_35'


    def parseCompilerOutput(self, output, expected):
        compilerLines = fnmatch.filter(output.splitlines(), '*):*warning*:*')
        warnings = set([cl[cl.index('warning:') + 8:].strip() for cl in compilerLines])

        # filter out ignored warnings
        warnings = CudaWarning.subtractWarningSets(warnings, self.getIgnoredWarnings())
        countDetected = len(warnings)

        if not hasattr(expected, '__iter__'): # this is not an iterable
            expected = [expected]
        countExpected = len(expected)

        unexpectedWarnings = CudaWarning.subtractWarningSets(warnings, expected)
        countUnexpected = len(unexpectedWarnings)

        if countDetected > 0 and countDetected == countExpected and countUnexpected == 0:
            self.testResults.setDefaultCompileResult(CudaTestResult.COMPILER_REPORTED)
        elif countDetected > countExpected or countUnexpected > 0:
            self.testResults.setDefaultCompileResult(CudaTestResult.COMPILER_UNEXPECTED)
            print 'WARNING: Unexpected compiler warning in {0}\n{1}'.format(
                    self.qualifiedProgram, '\n'.join(['\t'+h for h in unexpectedWarnings])
                )
        else:
            self.testResults.setDefaultCompileResult(CudaTestResult.COMPILER_NO_WARNINGS)


    def parseOutput(self, testResult, logText, expected, argsUsed):
        if not argsUsed: # don't print empty list
            argsUsed = ''

        if expected in logText:
            self.testResults.add(CudaTestResult.TOOL_REPORTED)
            return

        if testResult:
            if expected is CudaError.NO_ERR:
                self.testResults.add(CudaTestResult.TOOL_UNEXPECTED)
            else:
                self.testResults.add(CudaTestResult.TOOL_MISREPORTED)
            print 'WARNING: Unexpected output in', self.program, argsUsed, '\n\t'
            lines = logText.split('\n', 2)
            print lines[1] if len(lines) > 2 else lines
        else:
            self.testResults.add(CudaTestResult.TOOL_UNREPORTED)
            print 'WARNING: Unreported error in', self.program, argsUsed


    def runTestCases(self):
        if not os.path.isfile(self.qualifiedProgram):
            print 'ERROR: Skipping test, program', self.qualifiedProgram, 'not found'
            self.testResults.addSkipped(len(self.testCases))
            return

        for tc in self.testCases:
            logFileName = 'log/' + self.program + '_' + str(int(time()))
            profileToolArgs = [ # hardcode defaults in case they change 
                '--check-device-heap','yes',       # default
                '--check-api-memory-access','yes', # default
                '--leak-check','no',               # default
                '--report-api-errors','explicit',  # default
                '--error-exitcode', '1', 
                '--log-file', logFileName,
                '--tool', self.testMode,
            ]
            if self.testMode == CudaTestMode.RACECHECK:
                profileToolArgs += ['--racecheck-report','hazard']
            programArgs = tc.getArgsList()

            testExitCode = call([self.profileTool] + profileToolArgs + [self.qualifiedProgram] + programArgs)

            if not os.path.isfile(self.qualifiedProgram):
                self.skipped += 1
                print 'ERROR: Log file', logFileName, 'not found, skipping test case'
                continue

            with open(logFileName) as logFile:
                # this will be expensive if log files get big, but should be faster for our small tests
                log = logFile.read()
            
            self.parseOutput(testExitCode, log, tc.getExpectedText(), programArgs)



class CudaRaceTestProgram(CudaTestProgram):


    def __init__(self, 
            sourceFile, 
            testCases, 
            compiler='nvcc', 
            profileTool='cuda-memcheck', 
            testMode=CudaTestMode.RACECHECK, 
            macros={},
            ):
        super(self.__class__, self).__init__(sourceFile, 
                testCases, 
                compiler, 
                profileTool, 
                testMode, 
                macros,
                )


    def getIgnoredWarnings(self):
        # ignore variable used before set and variable unused warnings in racey tests
        # because fixing them would introduce additional read/write dependencies and
        # over-complicate the tests
        return [CudaWarning.VAR_USED_BEFORE_SET, CudaWarning.VAR_UNUSED]


    def getTargetArchitecture(self):
        # this should be at least sm_35 for dynamic parallelism, but Amazon's K520s are only sm_35
        return 'sm_35'


    def parseOutput(self, testResult, logText, expected, argsUsed):
        hazardLines = fnmatch.filter(logText.splitlines(), '*[EWI][RAN][RF][ON]*:*Potential*hazard detected*')
        hazards = set([h[ h.index('Potential ') + 10 # len('Potential ')
                        : h.rindex('__') + 2 ] for h in hazardLines])
        countDetected = len(hazards)

        if expected is CudaError.NO_RACE:
            if not testResult and countDetected == 0 and CudaError.NO_RACE in logText:
                self.testResults.add(CudaTestResult.TOOL_REPORTED)
                return # no error
            else:
                expected = []

        if not hasattr(expected, '__iter__'): # this is not an iterable
            expected = [expected]
        
        unexpectedHazards = hazards.difference(expected)
        countExpected = len(expected)
        countUnexpected = len(unexpectedHazards)
       
        if not argsUsed: # don't print empty list
            argsUsed = ''
        msg = None

        if countDetected == countExpected:
            if countUnexpected > 0:
                self.testResults.add(CudaTestResult.TOOL_MISREPORTED)
                msg = 'Misreported error'
            else:
                self.testResults.add(CudaTestResult.TOOL_REPORTED)
        elif countDetected > countExpected:
            self.testResults.add(CudaTestResult.TOOL_UNEXPECTED)
            msg = 'Unexpected error'
        else:
            self.testResults.add(CudaTestResult.TOOL_UNREPORTED)
            msg = 'Unreported error'
            
        if msg is None:
            return

        print 'WARNING: {0} in {1} {2}'.format(msg, self.program, argsUsed)
        if unexpectedHazards:
            print '\n'.join(['\t'+h for h in unexpectedHazards])
