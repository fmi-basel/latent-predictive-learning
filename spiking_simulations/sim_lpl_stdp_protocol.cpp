/* 
* Copyright 2014-2022 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "auryn.h"
#include "LPLConnection.h"

namespace po = boost::program_options;

using namespace auryn;

int main(int ac, char* av[]) 
{

	double w = 0.2;
	double w_ext = w;
	double wmax = 1.0;

	double w_ee = w;
	double w_ei = w;

	double gamma = 1.0;
	double w_ie = gamma;
	double w_ii = gamma;

	NeuronID ne = 1;
	NeuronID ni = ne/4;


	double sparseness = 1.0;
	double lambda = 1.0;
	double phi = 1.0;

	bool quiet = false;
	bool verbose = false;

	double tau_chk = 100e-3;
	double simtime = 3600.;
	double primetime = 0.;
	double stimtime = simtime;
	double wmat_interval = 600.;

	double ampa_nmda_ratio = 1.0;
	double wstim = 0.1;

	NeuronID psize = 200;
	NeuronID plen = 3;

	std::string currentfile = "";

	double stimfreq = 2;
	double initial_mean_trace = 20;
	double initial_sigma2_trace = 0.0;

	double ampl = 1.0;
	bool adapt = false;
	bool ei_plastic = false;

	double bg_rate = 2;
	bool fast = false;

	double tau_mean = 10.;
	double eta = 1e-3;
	double onperiod = 2;
	double offperiod = 30;
	double scale = 1;

	double tau_sigma2 = 300.;

	int n_strengthen = 0;

	std::string dir = ".";
	std::string stimfile = "";
	std::string label = "";

	const char * file_prefix = "lpl";
	char strbuf [255];
	std::string msg;

	int seed = 123;
	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("quiet", "quiet mode")
            ("verbose", "verbose mode")
            ("seed", po::value<int>(), "random seed")
            ("load", po::value<std::string>(), "input weight matrix")
            ("eta", po::value<double>(), "learning rate")
            ("bgrate", po::value<double>(), "PoissonGroup external firing rate")
            ("sparseness", po::value<double>(), "overall network sparseness")
            ("tau_mean", po::value<double>(), "homeostatic time constant for mean estimate")
            ("tau_sigma2", po::value<double>(), "homeostatic time constant for variance estimate")
            ("lambda", po::value<double>(), "push strength")
            ("phi", po::value<double>(), "pull strength")
            ("simtime", po::value<double>(), "simulation time")
            ("primetime", po::value<double>(), "priming time")
            ("dir", po::value<std::string>(), "output dir")
            ("label", po::value<std::string>(), "output label")
            ("wext", po::value<double>(), "wext")
            ("wee", po::value<double>(), "wee")
            ("wei", po::value<double>(), "wei")
            ("wie", po::value<double>(), "wie")
            ("wii", po::value<double>(), "wii")
            ("wmax", po::value<double>(), "wmax")
            ("ampa", po::value<double>(), "ampa nmda ratio")
            ("ne", po::value<int>(), "no of exc units")
            ("stimfile", po::value<std::string>(), "stimulus ras file")
            ("wstim", po::value<double>(), "weight of stimulus connections")
            ("stimtime", po::value<double>(), "time of stimulus on")
            ("psize", po::value<int>(), "population size for correlated inputs")
            ("plen", po::value<int>(), "number of input populations")
            ("stimfreq", po::value<double>(), "CorrelatedPoissonGroup frequency default = 100")
            ("initmean", po::value<double>(), "Initialize mean trace")
            ("initsigma2", po::value<double>(), "Initialize sigma2 trace")
            ("adapt", "use adapting excitatory neurons")
            ("fast", "turn off some of the monitors to run faster")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
			std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("quiet")) {
			quiet = true;
        } 

        if (vm.count("verbose")) {
			verbose = true;
        } 

        if (vm.count("seed")) {
           std::cout << "seed set to " 
                 << vm["seed"].as<int>() << ".\n";
			seed = vm["seed"].as<int>();
        } 


        if (vm.count("eta")) {
			std::cout << "eta set to " 
                 << vm["eta"].as<double>() << ".\n";
			eta = vm["eta"].as<double>();
        } 


        if (vm.count("sparseness")) {
			std::cout << "sparseness set to " 
                 << vm["sparseness"].as<double>() << ".\n";
			sparseness = vm["sparseness"].as<double>();
        } 

        if (vm.count("tau_mean")) {
			std::cout << "tau_mean set to " 
                 << vm["tau_mean"].as<double>() << ".\n";
			tau_mean = vm["tau_mean"].as<double>();
        } 

        if (vm.count("tau_sigma2")) {
			std::cout << "tau_sigma2 set to " 
                 << vm["tau_sigma2"].as<double>() << ".\n";
			tau_sigma2 = vm["tau_sigma2"].as<double>();
        } 

        if (vm.count("lambda")) {
			std::cout << "lambda set to " 
                 << vm["lambda"].as<double>() << ".\n";
			lambda = vm["lambda"].as<double>();
        } 

        if (vm.count("phi")) {
			std::cout << "phi set to " 
                 << vm["phi"].as<double>() << ".\n";
			phi = vm["phi"].as<double>();
        } 

        if (vm.count("simtime")) {
			std::cout << "simtime set to " 
                 << vm["simtime"].as<double>() << ".\n";
			simtime = vm["simtime"].as<double>();
			stimtime = simtime;
        } 

        if (vm.count("primetime")) {
			std::cout << "primetime set to " 
                 << vm["primetime"].as<double>() << ".\n";
			primetime = vm["primetime"].as<double>();
        } 

        if (vm.count("dir")) {
			std::cout << "dir set to " 
                 << vm["dir"].as<std::string>() << ".\n";
			dir = vm["dir"].as<std::string>();
        } 

        if (vm.count("label")) {
			std::cout << "label set to " 
                 << vm["label"].as<std::string>() << ".\n";
			label = vm["label"].as<std::string>();
        } 

        if (vm.count("wext")) {
			std::cout << "wext set to " 
                 << vm["wext"].as<double>() << ".\n";
			w_ext = vm["wext"].as<double>();
        } 


        if (vm.count("wee")) {
			std::cout << "wee set to " 
                 << vm["wee"].as<double>() << ".\n";
			w_ee = vm["wee"].as<double>();
        } 

        if (vm.count("wei")) {
			std::cout << "wei set to " 
                 << vm["wei"].as<double>() << ".\n";
			w_ei = vm["wei"].as<double>();
        } 

        if (vm.count("wie")) {
			std::cout << "wie set to " 
                 << vm["wie"].as<double>() << ".\n";
			w_ie = vm["wie"].as<double>();
        } 

        if (vm.count("wii")) {
			std::cout << "wii set to " 
                 << vm["wii"].as<double>() << ".\n";
			w_ii = vm["wii"].as<double>();
        } 

        if (vm.count("wmax")) {
			std::cout << "wmax set to " 
                 << vm["wmax"].as<double>() << ".\n";
			wmax = vm["wmax"].as<double>();
        } 

        if (vm.count("ampa")) {
           std::cout << "ampa set to " 
                 << vm["ampa"].as<double>() << ".\n";
			ampa_nmda_ratio = vm["ampa"].as<double>();
        } 

        if (vm.count("ne")) {
           std::cout << "ne set to " 
                 << vm["ne"].as<int>() << ".\n";
			ne = vm["ne"].as<int>();
			ni = ne/4;
        } 

        if (vm.count("stimfile")) {
           std::cout << "stimfile set to " 
                 << vm["stimfile"].as<std::string>() << ".\n";
			stimfile = vm["stimfile"].as<std::string>();
        } 

        if (vm.count("wstim")) {
           std::cout << "wstim set to " 
                 << vm["wstim"].as<double>() << ".\n";
			wstim = vm["wstim"].as<double>();
        } 

        if (vm.count("stimtime")) {
           std::cout << "stimtime set to " 
                 << vm["stimtime"].as<double>() << ".\n";
			stimtime = vm["stimtime"].as<double>();
        } 

        if (vm.count("psize")) {
           std::cout << "psize set to " 
                 << vm["psize"].as<int>() << ".\n";
			psize = vm["psize"].as<int>();
        } 

        if (vm.count("plen")) {
           std::cout << "plen set to " 
                 << vm["plen"].as<int>() << ".\n";
			plen = vm["plen"].as<int>();
        } 

        if (vm.count("stimfreq")) {
           std::cout << "stimfreq set to " 
                 << vm["stimfreq"].as<double>() << ".\n";
			stimfreq = vm["stimfreq"].as<double>();
        } 

        if (vm.count("initmean")) {
           std::cout << "initmean set to " 
                 << vm["initmean"].as<double>() << ".\n";
			initial_mean_trace = vm["initmean"].as<double>();
        } 

        if (vm.count("initsigma2")) {
           std::cout << "initsigma2 set to " 
                 << vm["initsigma2"].as<double>() << ".\n";
			initial_sigma2_trace = vm["initsigma2"].as<double>();
        } 

        if (vm.count("chk")) {
           std::cout << "chk set to " 
                 << vm["chk"].as<double>() << ".\n";
			tau_chk = vm["chk"].as<double>();
        } 

        if (vm.count("adapt")) {
           std::cout << "adaptation on " << std::endl;
			adapt = true;
        } 

        if (vm.count("fast")) {
           std::cout << "fast on " << std::endl;
			fast = true;
        } 

        if (vm.count("eiplastic")) {
           std::cout << "eiplastic on " << std::endl;
			ei_plastic = true;
        } 
    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }




	auryn_init(ac, av, dir, file_prefix);
	sys->quiet = quiet;
	sys->set_master_seed(seed);

	logger->set_logfile_loglevel( PROGRESS );
	if ( verbose ) logger->set_logfile_loglevel( EVERYTHING );

	// TODO 
	// Need to add current injector
	// Need to replace Poisson input groups with one file input group
	//


	logger->msg("Setting up neuron groups ...",PROGRESS,true);


	NeuronGroup * neurons_e;
	if ( adapt ) {
		neurons_e = new AIFGroup(ne);
		((AIFGroup*)neurons_e)->set_ampa_nmda_ratio(ampa_nmda_ratio);
		((AIFGroup*)neurons_e)->dg_adapt1=1.0;
	} else {
		neurons_e = new IFGroup(ne);
		((IFGroup*)neurons_e)->set_ampa_nmda_ratio(ampa_nmda_ratio);
	}
	// IFGroup * neurons_i = new IFGroup(ni);

	// initialize membranes
	neurons_e->random_mem(-60e-3,10e-3);
	// neurons_i->set_tau_mem(10e-3);
	// neurons_i->random_mem(-60e-3,10e-3);

	// ((IFGroup*)neurons_i)->set_ampa_nmda_ratio(ampa_nmda_ratio);




	std::stringstream oss;
	oss << "Activating File input group ... ";
	logger->msg(oss.str(), PROGRESS, true);
	SpikingGroup * input_group = new FileInputGroup(psize, "spikes.ras", false, 10.0); // last arguments enable loop mode with 10s delay

	logger->msg("Adding voltage clamp",PROGRESS,true);
	VoltageClamp * curr_inject = new VoltageClamp(neurons_e, "post_voltage.dat");
	curr_inject->loop = false;
	curr_inject->set_loop_grid(10.0);


	// Use the same time constant for the online rate estimate in the progress bar
	sys->set_online_rate_monitor_id(0);
	sys->set_online_rate_monitor_tau(tau_chk);


	// msg = "Setting up I connections ...";
	// logger->msg(msg,PROGRESS,true);
	// SparseConnection * con_ie = new SparseConnection(neurons_i, neurons_e, w_ie, sparseness, GABA);
	// SparseConnection * con_ii = new SparseConnection(neurons_i, neurons_i, w_ii, sparseness, GABA);

	msg =  "Setting up E connections ...";
	logger->msg(msg,PROGRESS,true);
	// SparseConnection * con_ei = new SparseConnection(neurons_e, neurons_i, w_ei, sparseness, GLUT);


	msg = "Initializing traces ...";
	logger->msg(msg,PROGRESS,true);


	// RateChecker * chk = new RateChecker( neurons_e , -0.1 , 20.*lambda , tau_chk);
	// To add a weight checker uncomment the following line
	// WeightChecker * wchk = new WeightChecker( con_ee, 0.159, 0.161 );



	BinarySpikeMonitor * smon_input = new BinarySpikeMonitor( input_group, sys->fn("input_group","spk") );
	// DelayedSpikeMonitor * dsmon_input = new DelayedSpikeMonitor( input_group, sys->fn("input_group","ras") );

	// init plastic input connection
	LPLConnection * con_lpl = new LPLConnection(input_group, neurons_e, w_ext, sparseness, eta, lambda, phi, tau_mean);
	con_lpl->set_tau_weight_decay(1e9); // 30min weight decay time constant
	con_lpl->use_weight_decay = false;
	bool augment_gradient = false;
	double tau_rms = 100.0;
	con_lpl->delta = 0.0e-5;
	con_lpl->augment_gradient = augment_gradient;
	con_lpl->set_tau_rms(tau_rms);
	con_lpl->set_max_weight(wmax);
	con_lpl->tr_post_mean->set_all(initial_mean_trace*con_lpl->tr_post_mean->get_tau());
	con_lpl->tr_post_sigma2->set_all(initial_sigma2_trace);
	// con_lpl->tr_post_sigma2->set_all(0.25);
	// con_lpl->tr_post_mean->set_all(35.0*con_lpl->tr_post_mean->get_tau());
	
	StateMonitor * smon = new StateMonitor( con_lpl->tr_post_sigma2, 0, sys->fn("con_lpl","sig2"), 0.1 ); 

	WeightMonitor * wmon = new WeightMonitor( con_lpl, sys->fn("input_con","syn"), 0.1 ); 
	wmon->add_equally_spaced(50);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	BinarySpikeMonitor * smon_e = new BinarySpikeMonitor( neurons_e, sys->fn("exc","spk"), 2500);
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, sys->fn("exc","prate"), 0.25 );

	if (primetime>0) {
		msg = "Priming ...";
		logger->msg(msg,PROGRESS,true);
		con_lpl->stdp_active = false;
		sys->run(primetime,true);
	}


	logger->msg("Simulating ...",PROGRESS,true);
	con_lpl->stdp_active = true;

	if (!sys->run(simtime,true)) 
			errcode = 1;



	if (!fast) {
		logger->msg("Saving neurons state ...",PROGRESS,true);
		neurons_e->write_to_file(sys->fn("exc","nstate").c_str() );
		// neurons_i->write_to_file(sys->fn("inh","nstate").c_str() );

		logger->msg("Saving weight matrix ...",PROGRESS,true);
		con_lpl->write_to_file(sys->fn("input_con","wmat"));
	}

	if (errcode) {
		auryn_abort(errcode);
	}

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;
}
