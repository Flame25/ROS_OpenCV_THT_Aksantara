# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/gadzz/miniconda3/envs/opencv/bin/cmake

# The command to remove a file.
RM = /home/gadzz/miniconda3/envs/opencv/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/gadzz/Documents/ROS_OpenCV/Soal 8"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/gadzz/Documents/ROS_OpenCV/Soal 8/build"

# Include any dependencies generated for this target.
include CMakeFiles/THT_Soal8.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/THT_Soal8.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/THT_Soal8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/THT_Soal8.dir/flags.make

CMakeFiles/THT_Soal8.dir/main.cpp.o: CMakeFiles/THT_Soal8.dir/flags.make
CMakeFiles/THT_Soal8.dir/main.cpp.o: /home/gadzz/Documents/ROS_OpenCV/Soal\ 8/main.cpp
CMakeFiles/THT_Soal8.dir/main.cpp.o: CMakeFiles/THT_Soal8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/gadzz/Documents/ROS_OpenCV/Soal 8/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/THT_Soal8.dir/main.cpp.o"
	/home/gadzz/miniconda3/envs/opencv/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/THT_Soal8.dir/main.cpp.o -MF CMakeFiles/THT_Soal8.dir/main.cpp.o.d -o CMakeFiles/THT_Soal8.dir/main.cpp.o -c "/home/gadzz/Documents/ROS_OpenCV/Soal 8/main.cpp"

CMakeFiles/THT_Soal8.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/THT_Soal8.dir/main.cpp.i"
	/home/gadzz/miniconda3/envs/opencv/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/gadzz/Documents/ROS_OpenCV/Soal 8/main.cpp" > CMakeFiles/THT_Soal8.dir/main.cpp.i

CMakeFiles/THT_Soal8.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/THT_Soal8.dir/main.cpp.s"
	/home/gadzz/miniconda3/envs/opencv/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/gadzz/Documents/ROS_OpenCV/Soal 8/main.cpp" -o CMakeFiles/THT_Soal8.dir/main.cpp.s

# Object files for target THT_Soal8
THT_Soal8_OBJECTS = \
"CMakeFiles/THT_Soal8.dir/main.cpp.o"

# External object files for target THT_Soal8
THT_Soal8_EXTERNAL_OBJECTS =

THT_Soal8: CMakeFiles/THT_Soal8.dir/main.cpp.o
THT_Soal8: CMakeFiles/THT_Soal8.dir/build.make
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_gapi.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_stitching.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_alphamat.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_aruco.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_bgsegm.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_bioinspired.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_ccalib.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_cvv.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_dnn_objdetect.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_dnn_superres.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_dpm.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_face.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_freetype.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_fuzzy.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_hdf.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_hfs.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_img_hash.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_intensity_transform.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_line_descriptor.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_mcc.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_quality.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_rapid.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_reg.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_rgbd.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_saliency.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_stereo.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_structured_light.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_superres.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_surface_matching.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_tracking.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_videostab.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_wechat_qrcode.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_xfeatures2d.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_xobjdetect.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_xphoto.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_shape.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_highgui.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_datasets.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_plot.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_text.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_ml.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_phase_unwrapping.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_optflow.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_ximgproc.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_video.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_videoio.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_imgcodecs.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_objdetect.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_calib3d.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_dnn.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_features2d.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_flann.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_photo.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_imgproc.so.4.9.0
THT_Soal8: /home/gadzz/miniconda3/envs/opencv/lib/libopencv_core.so.4.9.0
THT_Soal8: CMakeFiles/THT_Soal8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/gadzz/Documents/ROS_OpenCV/Soal 8/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable THT_Soal8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/THT_Soal8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/THT_Soal8.dir/build: THT_Soal8
.PHONY : CMakeFiles/THT_Soal8.dir/build

CMakeFiles/THT_Soal8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/THT_Soal8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/THT_Soal8.dir/clean

CMakeFiles/THT_Soal8.dir/depend:
	cd "/home/gadzz/Documents/ROS_OpenCV/Soal 8/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/gadzz/Documents/ROS_OpenCV/Soal 8" "/home/gadzz/Documents/ROS_OpenCV/Soal 8" "/home/gadzz/Documents/ROS_OpenCV/Soal 8/build" "/home/gadzz/Documents/ROS_OpenCV/Soal 8/build" "/home/gadzz/Documents/ROS_OpenCV/Soal 8/build/CMakeFiles/THT_Soal8.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/THT_Soal8.dir/depend
