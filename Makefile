CXX = g++
MEX = mex
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -fPIC
MEXCXXFLAGS = -std=c++11 -Wall -Wextra -O3
GTESTDIR = /usr/src/gtest

PCST_SRCS = src/pcst_fast.cc src/pcst_fast.h src/pairing_heap.h src/priority_queue.h

main_dimacs: src/main_dimacs.cc $(PCST_SRCS)
	$(CXX) $(CXXFLAGS) -o main_dimacs src/main_dimacs.cc src/pcst_fast.cc


# Unit tests
run_tests: run_pcst_fast_test

# Google Test framework
gtest-all.o: $(GTESTDIR)/src/gtest-all.cc
	$(CXX) $(CXXFLAGS) -I $(GTESTDIR) -c -o $@ $<

pcst_fast_test: gtest-all.o $(PCST_SRCS) src/pcst_fast_test.cc
	$(CXX) $(CXXFLAGS) -c -o pcst_fast.o src/pcst_fast.cc
	$(CXX) $(CXXFLAGS) -c -o pcst_fast_test.o src/pcst_fast_test.cc
	$(CXX) $(CXXFLAGS) -o pcst_fast_test gtest-all.o pcst_fast.o pcst_fast_test.o -pthread

run_pcst_fast_test: pcst_fast_test
	./pcst_fast_test
