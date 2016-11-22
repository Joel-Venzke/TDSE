CC=g++
CFLAGS=-c -Wall -std=c++11
LDFLAGS=
SOURCES=Hamiltonian.cpp Parameters.cpp Simulation.cpp Wavefuction.cpp Observables.cpp Pulse.cpp TDSE.cpp json.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=../bin/TDSE

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm ${OBJECTS} ${EXECUTABLE}
