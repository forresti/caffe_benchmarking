#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = main.o  helpers.o
EXENAME = main

#for Titan:
#CC = CC
#CCOPTS = -c -O3 -g -I$(MPICH_DIR)/include
#LINK = CC

#For Firebox (fbox.mill):
CC = mpiCC
CCOPTS = -c -O3 -g 
LINK = mpiCC
LINKOPTS =
 
all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

