#-----------------------------*-python-*----------------------------------------#
# file   config/application_unit_test.py
# author Alex Long <along@lanl.gov>
# date   Monday, August 12, 2016, 5:44 pm
# brief  Provide a python class that aids in creating unit tests that run
#        interactive user codes (i.e.: run a binary that reads an
#        input file and diff the resulting output file).
# note   Copyright (C) 2016, Los Alamos National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#

import sys
import os
import re
import subprocess

################################################################################
# function that returns the path of the input string, if found
def which(program):
  def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

  fpath, fname = os.path.split(program)
  if fpath:
    if is_exe(program):
      return program
  else:
    for path in os.environ["PATH"].split(os.pathsep):
      path = path.strip('"')
      exe_file = os.path.join(path, program)
      if is_exe(exe_file):
        return exe_file

    return None
################################################################################

################################################################################
# function to set flags for aprun
def set_aprun_depth_flags(numPE, mpi_cores_per_cpu):
  depth = mpi_cores_per_cpu/ numPE
  remainder = mpi_cores_per_cpu % numPE
  if (remainder > 0):
    print("Expecting the requested number of ranks ({0}) to be a factor of the \
      ranks/node ({1})".format(numPE, mpi_cores_per_cpu))
    print_final_fail_msg("TEST")
    sys.exit(1)
  return "-d {0}".format(depth)
################################################################################

##############################################################################
# print unit test footer fail message (this is what CMake looks for to
# indicate failure, it does not look for a non-zero return code!)
def print_final_fail_msg(testname):
  print("*****************************************************************")
  print("**** {0}: FAILED.".format(testname))
  print("*****************************************************************")
##############################################################################

################################################################################
# search string with regular expression and return the first matching component
def simple_search(regex, arg_string):
  return_str = ""
  if (regex.search(arg_string)):
    return_str = regex.findall(arg_string)[0]
  else:
    return_str = "not_found"
  return return_str
################################################################################

################################################################################
# print contents of file
def print_file(file_name):
  f_temp = open(file_name, 'r')
  for line in f_temp.readlines():
    print(line.strip())
  f_temp.close()
################################################################################

################################################################################
# check to see if the varaible name was set with a value in string (copies CMake
# for "if(<variable>)" logic
def is_set(param_string):
  return_bool = False
  if (param_string != "" and param_string != "not_found"):
    return_bool = True
  return return_bool
################################################################################

################################################################################
# check to see if the variable name was found in the string (copies CMake logic
# for DEFINED in CMake)
def is_defined(param_string):
  return_bool = False
  if (param_string != "not_found"):
    return_bool = True
  return return_bool
################################################################################

################################################################################
# Class encapsulating all functions needed for unit testing
class UnitTest:
  re_app = re.compile("APP=([^\s]*)")
  #re_build_env = re.compile("BUILDENV=([^\s]*)")
  re_gold_in_file = re.compile("GOLDFILE=([^\s]*)")
  re_std_in_file = re.compile("STDINFILE=([^\s]*)")
  #re_test_args = re.compile("TEST_ARGS=([^\s]*)")
  #re_pass_regex = re.compile("PASS_REGEX=([^\s]*)")
  #re_fail_regex = re.compile("FAIL_REGEX=([^\s]*)")
  #re_pe_list = re.compile("PE_LIST=([^\s]*)")
  re_project_binary_dir = re.compile("PROJECT_BINARY_DIR=([^\s]*)")
  re_project_source_dir = re.compile("PROJECT_SOURCE_DIR=([^\s]*)")
  re_test_name = re.compile("TESTNAME=([^\s]*)")
  re_numPE = re.compile("numPE=([^\s]*)")
  re_mpiexec = re.compile("MPIEXEC=([^\s]*)")
  re_mpi_cores_per_cpu = re.compile("MPI_CORES_PER_CPU=([^\s]*)")
  re_draco_info = re.compile("DRACO_INFO=([^\s]*)")
  re_run_cmd = re.compile("RUN_CMD=([^\s]*)")
  re_arg_value = re.compile("ARGVALUE=([^\s]*)")
  re_workdir = re.compile("WORKDIR=([^\s]*)")
  re_host_system_processor = re.compile("CMAKE_HOST_SYSTEM_PROCESSOR=([^\s]*)")

  def __init__(self):

    try:
      # get all command line arguments into a parseable string
      self.full_arg_string = ""
      for arg in sys.argv:
        self.full_arg_string = "{0} {1}".format(self.full_arg_string,arg)

      # setup and sanity check
      self.app = simple_search(self.re_app, self.full_arg_string)
      if not is_set(self.app):
        self.fatal_error("You must provide a value for APP")

      self.app = os.path.abspath(self.app)

      # set paths of input, binary directory and gold
      self.input = simple_search(self.re_std_in_file, self.full_arg_string)
      if is_set(self.input):
        self.input = os.path.abspath(self.input)
        if (not os.path.exists(self.input)):
          self.fatal_error("File not found, STDINFILE={0}.".format(self.input))

      self.bindir = os.path.abspath(self.app)

      self.gold = simple_search(self.re_gold_in_file, self.full_arg_string)
      if is_set(self.gold):
        if (not os.path.exists(self.gold)):
          self.fatal_error("File not found, GOLDFILE={0}.".format(self.gold))

      # Base the output file name off the name of the gold, if set.
      if is_set(self.gold):
        self.outfile = os.path.basename(self.gold)
      else:
        self.outfile = os.path.basename(self.app)

      self.project_binary_dir = simple_search(self.re_project_binary_dir, self.full_arg_string)
      self.project_source_dir = simple_search(self.re_project_source_dir, self.full_arg_string)
      self.errfile = "{0}/{1}.err".format(self.project_binary_dir, self.outfile)
      self.outfile = "{0}/{1}.out".format(self.project_binary_dir, self.outfile)

      if (not os.path.exists(self.app)):
        self.fatal_error("Cannot find {0}".format(self.app))
      else:
        print("Testing {0}".format(self.app))

      # Initialize number of passes and fails to zero.
      self.numpasses = 0
      self.numfails = 0

      debug = True
      if (debug):
        print("Running with the following parameters")
        print("   APP       = {0}".format(self.app))
        print("   BINDIR  = {0}".format(self.bindir))
        print("   PROJECT_BINARY_DIR = {0}".format(self.project_binary_dir))
        print("   OUTFILE   = {0}".format(self.outfile))
        print("   ERRFILE   = {0}".format(self.errfile))
        if (self.input):
          print("   STDINFILE = {0}".format(self.input))
        if (self.gold):
          print("   GOLDFILE = {0}".format(self.gold))

      # get the needed variables from the argument string using regex
      self.testname = simple_search(self.re_test_name, self.full_arg_string)
      self.numPE = simple_search(self.re_numPE, self.full_arg_string)
      self.mpi_cores_per_cpu = simple_search(self.re_mpi_cores_per_cpu, \
        self.full_arg_string)
      self.mpiexec = simple_search(self.re_mpiexec, self.full_arg_string)
      self.draco_info = simple_search(self.re_draco_info, self.full_arg_string)
      self.run_cmd = simple_search(self.re_run_cmd, self.full_arg_string)
      self.arg_value = simple_search(self.re_arg_value, self.full_arg_string)
      self.workdir = simple_search(self.re_workdir, self.full_arg_string)
      self.host_system_processor = simple_search(self.re_host_system_processor, \
        self.full_arg_string)

      # make dictionary of argument values for simple mapping between
      # cmake commands and python functions
      self.cmake_args = {"APP":self.app, "TESTNAME":self.testname, \
        "STDINFILE":self.input, "GOLDFILE":self.gold, \
        "PROJECT_BINARY_DIR":self.project_binary_dir, \
        "PROJECT_SOURCE_DIR":self.project_source_dir, "TESTNAME":self.testname, \
        "numPE":self.numPE, "MPI_CORES_PER_CPU":self.mpi_cores_per_cpu, \
        "MPIEXEC":self.mpiexec, "DRACO_INFO":self.draco_info, \
        "RUN_CMD":self.run_cmd, "ARGVALUE":self.arg_value, \
        "WORKDIR":self.workdir, \
        "CMAKE_HOST_SYSTEM_PROCESSOR":self.host_system_processor}

      # set endinaness for this test
      self.little_endian = True
      if (self.host_system_processor =="powerpc64") or \
          (self.host_system_processor=="ppc64"):
        self.litle_endian = False

      if is_set(self.mpi_cores_per_cpu):
        if self.mpi_cores_per_cpu == 0:
          self.fatal_error("Must set a nonzero number for MPI_CORES_PER_CPU")
        else:
          self.mpi_cores_per_cpu = int(self.mpi_cores_per_cpu)

      # Default filenames for output and error streams, add process IDs
      # to filenames to avoid errors when multiple processors run jobs
      self.testout = "testout_{0}".format(os.getpid())
      self.testerror = "testerror_{0}".format(os.getpid())

      # Look for numdiff in $PATH
      self.numdiff_exe = which("numdiff")
      if (not self.numdiff_exe):
        self.fatal_error("Numdiff not found in PATH")
      if (debug):
        print("   exenumdiff = {0}".format(self.numdiff_exe))

    except Exception:
      print("Caught exception: {0}  {1}".format( sys.exc_info()[0], \
        sys.exc_info()[1]))
      self.fatal_error("Ending test execution after catching exception")
  ##############################################################################

  ##############################################################################
  # Run the application and capture the output.
  def aut_runTests(self):

    try:
      print("\n=============================================")
      print("=== {0}".format(self.testname))
      print("=============================================")

      # open temporary files to redirect draco_info
      f_err = open(self.testerror, 'w')
      f_out = open(self.testout, 'w')
      # set runcmd for capsaicin
      self.run_cmd = self.run_cmd.replace(" ", ";")
      run_cmd_capsaicin = self.run_cmd

      # run draco --version with correct run command
      aprun_depth_options = ""
      draco_info_numPE = ""
      if is_set(self.numPE):
        # Use 1 proc to run draco_info
        draco_info_numPE = 1
        if self.mpiexec == "aprun":
          # Run with 1 proc, but tell aprun that we need the whole node.
          aprun_depth_options = set_aprun_depth_flags(1, self.mpi_cores_per_cpu)

      if (os.path.exists(self.draco_info)):
        testres = subprocess.call(["{0} {1} {2} {3} --version".format( \
          self.run_cmd, draco_info_numPE, aprun_depth_options, self.draco_info)], \
          stdout=f_out, stderr=f_err, shell=True)
        f_out.close()
        f_err.close()
        print(testres)
        if (testres != 0):
          print("Unable to run \'{0} {1} {2} {3} --version\'".format(self.run_cmd, \
            draco_info_numPE, aprun_depth_options, self.draco_info))
        else:
          f_out = open(self.testout, 'r')
          print(f_out.readlines())

      # add numPE to the output file
      safe_arg_value = ""
      if is_set(self.numPE):
        self.outfile = self.outfile.replace(".out", "-{0}.out".format(self.numPE))
        self.errfile = self.errfile.replace(".err", "-{0}.err".format(self.numPE))

      # clean up arg value
      if is_set(self.arg_value):
        safe_arg_value = self.arg_value.replace("[-]","")
        self.outfile = self.outfile.replace(".out", "-{0}.out".format(\
          safe_arg_value))
        self.errfile = self.errfile.replace(".err", "-{0}.err".format(\
          safe_arg_value))

      if ( self.mpiexec == "aprun"):
        # Run with requested number of processors, but tell aprun that we need the
        # whole node.
        if (self.numPE):
          aprun_depth_options = set_aprun_depth_flags(int(self.numPE), \
            self.mpi_cores_per_cpu)

      # go back to space separated list for run_cmd
      if is_defined(self.run_cmd):
        run_cmd_string = self.run_cmd.replace(";", " ")
        print(">>> Running: {0} {1} {2}".format(run_cmd_string, self.numPE,\
          aprun_depth_options))
        print(">>>          {0}".format(self.app))
        if (self.arg_value):
          print(">>>          {0}".format(self.arg_value))
      else:
        print(">>> Running: {0} {1}".format(self.app, self.arg_value))

      # use command line input style for MICs
      use_cl_input = False
      if os.path.exists(self.input):
        if (self.run_cmd == "run_test_on_mic"):
          arg_value = "{0} {1}".format(arg_value, "< {0}".format(self.input))
          use_cl_input = True
        print(">>>          < {0}".format(self.input))
      print(">>>          > {0}\n".format(self.outfile))

      # Run the application capturing all output.
      stdin_file = is_set(self.input)
      arg_value = self.arg_value.replace(" ", ";")
      f_out = open(self.testout, 'w')
      f_err = open(self.testerror, 'w')
      if stdin_file:
        f_in = open(self.input, 'r')


      # use the subprocess call to set the standard input
      if (stdin_file and not use_cl_input):
        testres = subprocess.call(["{0} {1} {2} {3} {4}".format(self.run_cmd, self.numPE, \
          aprun_depth_options, self.app, self.arg_value)], stdout=f_out, stdin=f_in, \
          stderr=f_err, shell=True)
      # using command line input mode with an stdin file puts the stdin file with
      # the argvalue string, if not stdin file is given the arg_value will not be
      # modified
      else:
        testres = subprocess.call(["{0} {1} {2} {3} {4}".format(self.run_cmd, self.numPE, \
          aprun_depth_options, self.app, self.arg_value)], stdout=f_out, \
          stderr=f_err, shell=True)

      # close file handles
      f_out.close()
      f_err.close()
      if (stdin_file): f_in.close();

      # go back to space delimited argument list
      if is_set(arg_value):
        arg_value = arg_value.replace(";", " ")

      # Now write the cleaned up file.
      f_clean = open(self.outfile, 'w')
      f_out = open(self.testout, 'r')
      re_aprun_remove = re.compile("Application [0-9]* resources: utime")
      for line in f_out.readlines():
        if(not re_aprun_remove.search(line)):
          f_clean.write(line)
        print(line.strip())
      f_clean.close()
      f_out.close()

      if (testres):
        # get last line written to stderror
        f_error = open(self.testerror)
        error_lines = f_error.readlines()
        last_error = error_lines.pop()
        print("Test FAILED:\n last message written to stderr: \'{0}".format(last_error))
        self.fatal_error("See {0} for full details.".format(self.outfile))
        f_error.close()
      else:
        print_file(self.outfile)
        self.passmsg("Application ran to completion")

      # cleanup temporary files
      os.remove(self.testout)
      os.remove(self.testerror)

    except Exception:
      print("Caught exception: {0}  {1}".format( sys.exc_info()[0], \
        sys.exc_info()[1]))
      self.fatal_error("Ending test execution after catching exception")
  ##############################################################################

  ##############################################################################
  # check to see if the output file contains a given string
  def output_contains(self, search_string):

    # make regular expression from input string
    re_search = re.compile(search_string)

    # search file for string
    f_clean = open(self.outfile)
    return_bool = False
    if (re_search.search(f_clean.read())):
      return_bool = True
    return return_bool
  ##############################################################################

  ##############################################################################
  # get a value with REGEX, see if it matches reference values
  def output_contains_value(self, search_regex, reference_value):
    # search file for string
    f_clean = open(self.outfile)
    return_bool = False
    if (search_regex.search(f_clean.read())):
      if (search_regex.findall(f_clean.read())[0] == reference_value):
        return_bool = True
    return return_bool
  ##############################################################################


  ##############################################################################
  # print unit test footer and output pass/fail messages
  def aut_report(self):
    print("*****************************************************************")
    if(self.numpasses and not self.numfails):
      print("**** {0}: PASSED.".format(self.testname))
    else:
      print("**** {0}: FAILED.".format(self.testname))
    print("*****************************************************************")
  ##############################################################################

  ##############################################################################
  # call numdiff between the gold and output file
  def aut_numdiff(self):

    try:
      # set numdiff run command
      numdiff_run_cmd = ""
      if is_defined(self.run_cmd):
        numdiff_run_cmd = self.run_cmd
        numdiff_run_cmd = numdiff_run_cmd.replace(" ", ";")
        # handle run_cmd on Cray environments
        if (self.mpiexec == "aprun" or self.mpiexec == "mpiexec"):
          numdiff_run_cmd = ""
        elif is_set(self.numPE):
          numdiff_run_cmd = "{0};1".format(numdiff_run_cmd)
      pretty_run_cmd = numdiff_run_cmd.replace(";", " ")

      # run numdiff command, redirecting stdout and stderr, get a unique
      # filename for the numdiff output and error files
      print("Comparing output to goldfile: ")
      print("{0} {1} \n {2} {3}".format(pretty_run_cmd, self.numdiff_exe, \
        self.outfile, self.gold))
      temp_numdiff_out = "numdiff_out_{0}".format(os.getpid())
      temp_numdiff_err = "numdiff_err_{0}".format(os.getpid())
      f_out = open( temp_numdiff_out, 'w')
      f_err = open( temp_numdiff_err, 'w')
      numdiff_res = subprocess.call(["{0} {1} {2} {3}".format(numdiff_run_cmd,
        self.numdiff_exe, self.outfile, self.gold)], shell=True,
        stdout=f_out, stderr=f_err)

      # close file handles
      f_out.close()
      f_err.close()

      # check return code of numdiff, if nonzero test fails
      if (not numdiff_res):
        self.passmsg("gold matches out.")
      else:
        self.failmsg("gold does not match out.")
        print("numdiff output = ")
        print_file(temp_numdiff_out)

      # cleanup temporary files
      os.remove(temp_numdiff_out)
      os.remove(temp_numdiff_err)

    except Exception:
      print("Caught exception: {0}  {1}".format( sys.exc_info()[0], \
        sys.exc_info()[1]))
      self.fatal_error("Ending test execution after catching exception")
  ##############################################################################

  ##############################################################################
  # call arbitrary diff command between two files
  def diff_two_files(self, cmake_dir_1, sub_path_1, cmake_dir_2, sub_path_2, \
      diff_name="numdiff"):

    try:
      if (not self.cmake_args.has_key(cmake_dir_1)) or \
          (not self.cmake_args.has_key(cmake_dir_2)):
        self.fatal_error("CMake arguments not speficied in command line")

      path_1 = "{0}/{1}".format(self.cmake_args[cmake_dir_1], sub_path_1)
      path_2 = "{0}/{1}".format(self.cmake_args[cmake_dir_2], sub_path_2)

      # check to make sure both files exist
      if (not os.path.exists(path_1)) or (not os.path.exists(path_2)):
        self.fatal_error("One or both filepaths do not exist")

      # set numdiff run command
      diff_run_cmd = ""
      if is_defined(self.run_cmd):
        diff_run_cmd = self.run_cmd
        diff_run_cmd = diff_run_cmd.replace(" ", ";")
        # handle run_cmd on Cray environments
        if (self.mpiexec == "aprun" or self.mpiexec == "mpiexec"):
          diff_run_cmd = ""
        elif is_set(self.numPE):
          diff_run_cmd = "{0};1".format(diff_run_cmd)
      pretty_run_cmd = diff_run_cmd.replace(";", " ")

      # Look for diff program in $PATH
      if (diff_name != "numdiff"):
        diff_exe = which(diff_name)
        if (not diff_exe):
          self.fatal_error("Diff command \"{0}\" not found in PATH".format( \
            diff_name))
      else:
        diff_exe = self.numdiff_exe

      # run diff command, redirecting stdout and stderr, get a unique
      # filename for the diff output and error files
      print("Comparing output of {0} and {1} with diff command: {2}".format( \
        path_1, path_2,  diff_exe))
      temp_diff_out = "diff_out_{0}".format(os.getpid())
      temp_diff_err = "diff_err_{0}".format(os.getpid())
      f_out = open(temp_diff_out, 'w')
      f_err = open(temp_diff_err, 'w')
      diff_res = subprocess.call(["{0} {1} {2} {3}".format(diff_run_cmd,
        diff_exe, path_1, path_2)], shell=True, stdout=f_out, stderr=f_err)

      # close file handles
      f_out.close()
      f_err.close()

      # check return code of numdiff, if nonzero test fails
      if (not diff_res):
        self.passmsg("two files match.")
      else:
        self.failmsg("two files differ.")
        print("dif output = ")
        print_file(temp_diff_out)

      # cleanup temporary files
      os.remove(temp_diff_out)
      os.remove(temp_diff_err)

    except Exception:
      print("Caught exception: {0}  {1}".format( sys.exc_info()[0], \
        sys.exc_info()[1]))
      self.fatal_error("Ending test execution after catching exception")
  ##############################################################################

  ##############################################################################
  #  print pass message and increment numpasses
  def passmsg(self, msg):
    print("Test Passes: {0}".format(msg))
    self.numpasses = self.numpasses+1
  ##############################################################################

  ##############################################################################
  #  print fail message and increment numfails
  def failmsg(self, msg):
    print("Test Fails: {0}".format(msg))
    self.numfails = self.numfails+1
  ##############################################################################

  ################################################################################
  # print string, print failing message (so ctest will interpret test as failure)
  # and exit with non-zero exit code
  def fatal_error(self, msg):
    self.numfails = self.numfails+1
    print(msg)
    print_final_fail_msg(self.testname)
    sys.exit(self.numfails)
  ################################################################################

  ##############################################################################
  # Checks the list of arguments for arg_name
  def check_arg_is_defined(self, arg_name):
    return ( self.full_arg_string.find(arg_name) != -1)
  ##############################################################################

  ##############################################################################
  # Checks the list of arguments for arg_name and make sure it is not empty
  def check_arg_is_set(self, arg_name):
    arg_regex = re.compile("{0}=([^\s]*)".format(arg_name))
    value = simple_search(arg_regex, self.full_arg_string)
    return is_set(value)
  ##############################################################################

  ##############################################################################
  # Checks the list of arguments for arg_name and then check arg_string's
  # value for check_string
  def check_arg_value(self, arg_name, check_string):
    arg_regex = re.compile("{0}=([^\s]*)".format(arg_name))
    arg_value = simple_search(arg_regex, self.full_arg_string)
    return ( arg_value.find(check_string) != -1)
  ##############################################################################

  ##############################################################################
  # Check to see if test is little endian
  def is_little_endian(self):
    return self.little_endian
  ##############################################################################

################################################################################
