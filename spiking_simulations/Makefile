# shorthand for your MPI C++ compiler
CC = mpicxx
# Your Auryn install path
AURYNDIR = $(HOME)/auryn/
# Path to libauryn.a (see compiling Auryn)
BUILDDIR = $(AURYNDIR)/build/release/

CFLAGS= -ansi -Wall -pipe -O3 -DNDEBUG -ffast-math -funsafe-math-optimizations \
        -march=native -mtune=native -pedantic \
        -I/usr/local/include -I$(AURYNDIR)/src -I$(AURYNDIR)/dev/src -I..

LDFLAGS=$(BUILDDIR)/src/libauryn.a \
		-L/usr/local/lib -lboost_program_options -lboost_serialization -lboost_mpi -lboost_system -lboost_filesystem

# # debug
# BUILDDIR = $(AURYNDIR)/build/debug/
# CFLAGS= -ansi -Wall -pipe -g \
#         -march=native -mtune=native -pedantic \
#         -I$(AURYNDIR)/src -I$(AURYNDIR)/dev/src
# 

OBJECTS=LPLConnection.o
SIMULATIONS=sim_lpl_spiking sim_lpl_stdp_protocol

.SECONDARY:

all: $(SIMULATIONS) aux

aux: local_inh_connectivity.wmat rates0.dat

rates0.dat: mk_rates.py
	python $<

local_inh_connectivity.wmat: mk_local_inh_connectivity.py
	python mk_local_inh_connectivity.py
	mv local_inh_connectivity.wmat.mtx local_inh_connectivity.wmat

sim_%: sim_%.o $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $< $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

%.o : ../%.cpp
	$(CC) $(CFLAGS) -c $<

figs: 

plot_%.tex: plot_%.gnu 
	gnuplot $< 
	epstopdf epslatex-inc.eps
	mv epslatex.out $@

%.pdf: %.tex
	pdflatex $< 

clean: 
	rm -f *.o $(SIMULATIONS) *.dat *.pdf *.png *.wmat *.ras
