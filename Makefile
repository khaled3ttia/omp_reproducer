default:
	clang++ -fopenmp -fopenmp-targets=nvptx64 -gline-tables-only -O3 main.cpp
