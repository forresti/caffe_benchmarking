#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = allreduce_benchmark.o  helpers.o
EXENAME = allreduce_benchmark

#for Titan:
#CC = CC
#CCOPTS = -c -O3 -g -I$(MPICH_DIR)/include
#LINK = CC

#For Firebox (fbox.mill):
CC = /opt/openmpi/bin/mpiCC
#CC=mpiCC
CCOPTS = -c -O3 -g 
#LINK=mpiCC
LINK = /opt/openmpi/bin/mpiCC
LINKOPTS =
 
all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

allreduce_benchmark.o : allreduce_benchmark.cpp helpers.h
	$(CC) $(CCOPTS) allreduce_benchmark.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

