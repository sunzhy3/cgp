# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szy/cgp_linux

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szy/cgp_linux/build

# Include any dependencies generated for this target.
include src/CMakeFiles/cgp_mat.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cgp_mat.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cgp_mat.dir/flags.make

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o: src/CMakeFiles/cgp_mat.dir/flags.make
src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o: ../src/cgp_mat.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/szy/cgp_linux/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o"
	cd /home/szy/cgp_linux/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o -c /home/szy/cgp_linux/src/cgp_mat.cpp

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cgp_mat.dir/cgp_mat.cpp.i"
	cd /home/szy/cgp_linux/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/szy/cgp_linux/src/cgp_mat.cpp > CMakeFiles/cgp_mat.dir/cgp_mat.cpp.i

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cgp_mat.dir/cgp_mat.cpp.s"
	cd /home/szy/cgp_linux/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/szy/cgp_linux/src/cgp_mat.cpp -o CMakeFiles/cgp_mat.dir/cgp_mat.cpp.s

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.requires:
.PHONY : src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.requires

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.provides: src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/cgp_mat.dir/build.make src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.provides.build
.PHONY : src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.provides

src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.provides.build: src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o

# Object files for target cgp_mat
cgp_mat_OBJECTS = \
"CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o"

# External object files for target cgp_mat
cgp_mat_EXTERNAL_OBJECTS =

src/libcgp_mat.a: src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o
src/libcgp_mat.a: src/CMakeFiles/cgp_mat.dir/build.make
src/libcgp_mat.a: src/CMakeFiles/cgp_mat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libcgp_mat.a"
	cd /home/szy/cgp_linux/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cgp_mat.dir/cmake_clean_target.cmake
	cd /home/szy/cgp_linux/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cgp_mat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cgp_mat.dir/build: src/libcgp_mat.a
.PHONY : src/CMakeFiles/cgp_mat.dir/build

src/CMakeFiles/cgp_mat.dir/requires: src/CMakeFiles/cgp_mat.dir/cgp_mat.cpp.o.requires
.PHONY : src/CMakeFiles/cgp_mat.dir/requires

src/CMakeFiles/cgp_mat.dir/clean:
	cd /home/szy/cgp_linux/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cgp_mat.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cgp_mat.dir/clean

src/CMakeFiles/cgp_mat.dir/depend:
	cd /home/szy/cgp_linux/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szy/cgp_linux /home/szy/cgp_linux/src /home/szy/cgp_linux/build /home/szy/cgp_linux/build/src /home/szy/cgp_linux/build/src/CMakeFiles/cgp_mat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cgp_mat.dir/depend
