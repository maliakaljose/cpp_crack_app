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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u16364/cpp_crack_app/build

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u16364/cpp_crack_app/build

# Include any dependencies generated for this target.
include CMakeFiles/Crack.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Crack.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Crack.dir/flags.make

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o: CMakeFiles/Crack.dir/flags.make
CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o: /home/u16364/cpp_crack_app/src/crack.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u16364/cpp_crack_app/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o -c /home/u16364/cpp_crack_app/src/crack.cpp

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u16364/cpp_crack_app/src/crack.cpp > CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.i

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u16364/cpp_crack_app/src/crack.cpp -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.s

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.requires:
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.requires

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.provides: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.requires
	$(MAKE) -f CMakeFiles/Crack.dir/build.make CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.provides.build
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.provides

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.provides.build: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o: CMakeFiles/Crack.dir/flags.make
CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o: /home/u16364/cpp_crack_app/src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u16364/cpp_crack_app/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o -c /home/u16364/cpp_crack_app/src/main.cpp

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u16364/cpp_crack_app/src/main.cpp > CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.i

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u16364/cpp_crack_app/src/main.cpp -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.s

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.requires:
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.requires

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.provides: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Crack.dir/build.make CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.provides

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.provides.build: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o: CMakeFiles/Crack.dir/flags.make
CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o: /home/u16364/cpp_crack_app/src/predict.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u16364/cpp_crack_app/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o -c /home/u16364/cpp_crack_app/src/predict.cpp

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u16364/cpp_crack_app/src/predict.cpp > CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.i

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u16364/cpp_crack_app/src/predict.cpp -o CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.s

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.requires:
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.requires

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.provides: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.requires
	$(MAKE) -f CMakeFiles/Crack.dir/build.make CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.provides.build
.PHONY : CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.provides

CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.provides.build: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o

# Object files for target Crack
Crack_OBJECTS = \
"CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o" \
"CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o" \
"CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o"

# External object files for target Crack
Crack_EXTERNAL_OBJECTS =

Crack: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o
Crack: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o
Crack: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o
Crack: CMakeFiles/Crack.dir/build.make
Crack: /glob/intel-python/python3/bin/../lib/libcaffe.so.1.1.0
Crack: /glob/intel-python/python3/lib/libopencv_xphoto.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_xobjdetect.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_ximgproc.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_xfeatures2d.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_tracking.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_text.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_surface_matching.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_structured_light.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_stereo.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_saliency.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_rgbd.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_reg.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_plot.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_optflow.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_line_descriptor.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_hdf.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_fuzzy.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_face.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_dpm.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_dnn.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_datasets.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_ccalib.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_bioinspired.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_bgsegm.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_aruco.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_videostab.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_videoio.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_video.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_superres.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_stitching.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_shape.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_photo.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_objdetect.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_ml.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_imgproc.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_imgcodecs.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_highgui.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_flann.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_features2d.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_core.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_calib3d.so.3.1.0
Crack: /glob/intel-python/versions/2018u2/intelpython3/lib/libboost_system.so.1.64.0
Crack: /glob/intel-python/python3/lib/libopencv_text.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_face.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_ximgproc.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_xfeatures2d.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_shape.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_video.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_objdetect.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_calib3d.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_features2d.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_ml.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_highgui.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_videoio.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_imgcodecs.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_imgproc.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_flann.so.3.1.0
Crack: /glob/intel-python/python3/lib/libopencv_core.so.3.1.0
Crack: CMakeFiles/Crack.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Crack"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Crack.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Crack.dir/build: Crack
.PHONY : CMakeFiles/Crack.dir/build

CMakeFiles/Crack.dir/requires: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/crack.cpp.o.requires
CMakeFiles/Crack.dir/requires: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/main.cpp.o.requires
CMakeFiles/Crack.dir/requires: CMakeFiles/Crack.dir/home/u16364/cpp_crack_app/src/predict.cpp.o.requires
.PHONY : CMakeFiles/Crack.dir/requires

CMakeFiles/Crack.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Crack.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Crack.dir/clean

CMakeFiles/Crack.dir/depend:
	cd /home/u16364/cpp_crack_app/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u16364/cpp_crack_app/build /home/u16364/cpp_crack_app/build /home/u16364/cpp_crack_app/build /home/u16364/cpp_crack_app/build /home/u16364/cpp_crack_app/build/CMakeFiles/Crack.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Crack.dir/depend
