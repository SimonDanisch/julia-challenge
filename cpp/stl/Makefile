main.o : main.cpp
		$(CXX) -O3 -march=native -mtune=native -std=c++17 main.cpp -o main.o
clean :
		rm main.o
benchmark : main.o
		./main.o
