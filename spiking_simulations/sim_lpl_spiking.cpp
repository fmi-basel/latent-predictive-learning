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

	double w_ee = 0;
	double w_ei = w;

	double gamma = 1.0;
	double w_ie = gamma;
	double w_ii = gamma;

	NeuronID ne = 100;
	NeuronID ni = 0;


	double sparseness = 0.2;
	double lambda = 1.0;
	double phi = 1.0;

	bool quiet = false;
	bool verbose = false;

	double tau_chk = 100e-3;
	double simtime = 3600.;
	double stimtime = simtime;
	double wmat_interval = 600.;

	double ampa_nmda_ratio = 1.0;

	NeuronID psize = 200;
	NeuronID plen = 3;

	std::string currentfile = "";

	bool adapt = false;
	bool ieplastic = false;
	bool augment_gradient = false;

	double bg_rate = 2;
	bool fast = false;

	double tau_mean = 10.;
	double tau_sigma2 = 300.;
	double eta = 1e-3; // exc learning rate
	double zeta = 1e-3; // inh learning rate

	double kappa = 3.0; // target firing rate for inhibitory plastiicty
	double tau_inh_stdp = 20e-3;

	double tau_weight_decay = 1e9; // basically no decay per default

	int n_strengthen = 0;

	std::string dir = ".";
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
            ("eta", po::value<double>(), "learning rate")
            ("zeta", po::value<double>(), "inh learning rate")
            ("kappa", po::value<double>(), "inh plast target firing rate")
            ("sparseness", po::value<double>(), "overall network sparseness")
            ("tau_mean", po::value<double>(), "homeostatic time constant for mean estimate")
            ("tau_sigma2", po::value<double>(), "homeostatic time constant for variance estimate")
            ("tau_weight_decay", po::value<double>(), "weight decay time constant")
            ("lambda", po::value<double>(), "push strength")
            ("phi", po::value<double>(), "pull strength")
            ("simtime", po::value<double>(), "simulation time")
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
            ("ni", po::value<int>(), "no of inh units")
            ("stimtime", po::value<double>(), "time of stimulus on")
            ("psize", po::value<int>(), "population size for correlated inputs")
            ("plen", po::value<int>(), "number of input populations")
            ("tau_chk", po::value<double>(), "checker time constant")
            ("adapt", "use adapting excitatory neurons")
            ("fast", "turn off some of the monitors to run faster")
            ("ieplastic", "make EI connection plastic")
            ("augment_gradient", "use gradient accelerated training")
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

        if (vm.count("zeta")) {
			std::cout << "zeta set to " 
                 << vm["zeta"].as<double>() << ".\n";
			zeta = vm["zeta"].as<double>();
        } 

        if (vm.count("kappa")) {
			std::cout << "kappa set to " 
                 << vm["kappa"].as<double>() << ".\n";
			kappa = vm["kappa"].as<double>();
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

        if (vm.count("tau_weight_decay")) {
			std::cout << "tau_weight_decay set to " 
                 << vm["tau_weight_decay"].as<double>() << ".\n";
			tau_weight_decay = vm["tau_weight_decay"].as<double>();
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
        } 

        if (vm.count("ni")) {
           std::cout << "ni set to " 
                 << vm["ni"].as<int>() << ".\n";
			ni = vm["ni"].as<int>();
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

        if (vm.count("tau_chk")) {
           std::cout << "tau_chk set to " 
                 << vm["tau_chk"].as<double>() << ".\n";
			tau_chk = vm["tau_chk"].as<double>();
        } 

        if (vm.count("adapt")) {
           std::cout << "adaptation on " << std::endl;
			adapt = true;
        } 

        if (vm.count("fast")) {
           std::cout << "fast on " << std::endl;
			fast = true;
        } 

        if (vm.count("ieplastic")) {
           std::cout << "ieplastic on " << std::endl;
			ieplastic = true;
        } 

        if (vm.count("augment_gradient")) {
           std::cout << "gradient accelerated training enabled" << std::endl;
			augment_gradient = true;
        } 
    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }


	double primetime = 3*tau_mean;


	auryn_init(ac, av, dir, file_prefix);
	sys->quiet = quiet;
	sys->set_master_seed(seed);

	logger->set_logfile_loglevel( PROGRESS );
	if ( verbose ) logger->set_logfile_loglevel( EVERYTHING );


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


	// initialize membranes
	neurons_e->random_mem(-60e-3,10e-3);
	// neurons_i->set_tau_mem(10e-3);
	// neurons_i->random_mem(-60e-3,10e-3);

	// ((IFGroup*)neurons_i)->set_ampa_nmda_ratio(ampa_nmda_ratio);




	std::stringstream oss;
	oss << "Activating inhom Poisson group input ... ";
	logger->msg(oss.str(), PROGRESS, true);

	std::vector<SpikingGroup*> list;
	for ( unsigned int i = 0 ; i<=plen ; ++i ) {
		std::sprintf(strbuf, "rates%i.dat",i);
		SpikingGroup * pgroup = new FileModulatedPoissonGroup(psize, strbuf, true); // last argument enables loop mode
		PopulationRateMonitor * pmon = new PopulationRateMonitor( pgroup, sys->fn("poisson",i,"prate"), 100 );
		list.push_back(pgroup);
	}
	ConcatGroup * input_group = new ConcatGroup(); // init meta input group
	input_group->set_name("input group");
	for ( unsigned int i = 0 ; i<=plen ; ++i ) {
		input_group->add_parent_group(list.at(i));
	}

	BinarySpikeMonitor * smon_input = new BinarySpikeMonitor( input_group, sys->fn("input_group","spk") );

	// Use the same time constant for the online rate estimate in the progress bar
	sys->set_online_rate_monitor_id(0);
	sys->set_online_rate_monitor_tau(tau_chk);


	// msg = "Initializing traces ...";
	// logger->msg(msg,PROGRESS,true);


	// RateChecker * chk = new RateChecker( neurons_e , -0.1 , 20.*lambda , tau_chk);
	// To add a weight checker uncomment the following line
	// WeightChecker * wchk = new WeightChecker( con_ee, 0.159, 0.161 );
	// DelayedSpikeMonitor * dsmon_input = new DelayedSpikeMonitor( input_group, sys->fn("input_group","ras") );
	
	msg =  "Setting up E connections ...";
	logger->msg(msg,PROGRESS,true);

	// init plastic input connection
	LPLConnection * con_lpl = new LPLConnection(input_group, neurons_e, w_ext, sparseness, eta, lambda, phi, tau_mean, tau_sigma2);
	con_lpl->use_weight_decay = false; 
	if ( tau_weight_decay < 1e5 ) {
		con_lpl->use_weight_decay = true; 
		con_lpl->set_tau_weight_decay(tau_weight_decay); 
	}

	// bool augment_gradient = true;
	double tau_rms = 100.0;
	con_lpl->augment_gradient = augment_gradient;
	con_lpl->set_tau_rms(tau_rms);
	con_lpl->set_max_weight(wmax);

	// record LPL dynamic variables
	const NeuronID rec_unit = 7;
	StateMonitor * statmon_lpl1 = new StateMonitor( con_lpl->tr_post_mean, rec_unit, sys->fn("tr_post_mean","dat"), 1.0 );
	StateMonitor * statmon_lpl2 = new StateMonitor( con_lpl->tr_post_sigma2, rec_unit, sys->fn("tr_post_sigma2","dat"), 1.0 );


	if ( w_ee > 0.0 ) {
		LPLConnection * con_lpl_rec = new LPLConnection(neurons_e, neurons_e, w_ext, sparseness, eta, lambda, phi, tau_mean, tau_sigma2);
		con_lpl_rec->use_weight_decay = false; 
		if ( tau_weight_decay < 1e5 ) {
			con_lpl_rec->use_weight_decay = true; 
			con_lpl_rec->set_tau_weight_decay(tau_weight_decay); 
		}

		con_lpl_rec->augment_gradient = augment_gradient;
		con_lpl_rec->set_tau_rms(tau_rms);
		con_lpl_rec->set_max_weight(wmax);
	}




	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	BinarySpikeMonitor * smon_e = new BinarySpikeMonitor( neurons_e, sys->fn("exc","spk"));
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, sys->fn("exc","prate"), 100 );

	SparseConnection * con_ei;
	SparseConnection * con_ie;
	if ( ni ) {
		msg =  "Setting up I neurons and connections ...";
		IFGroup * neurons_i = new IFGroup(ni);

		con_ei = new SparseConnection(neurons_e, neurons_i, w_ei, 0.2, AMPA); 
		con_ei->load_from_complete_file("local_inh_connectivity.wmat");
		// we assume the above mask is one or zero
		con_ei->scale_all(w_ei);
		con_ei->write_to_file(sys->fn("con_ei","wmat")); // store for debug reasons

		// con_ei = new LPLConnection(neurons_e, neurons_i, w_ei, 0.5, eta, lambda, phi, tau_mean);
		// con_ei->augment_gradient = augment_gradient;
		// con_ei->set_tau_rms(tau_rms);
		// con_ei->set_max_weight(wmax);

		if ( ieplastic ) {
			con_ie = new SymmetricSTDPConnection(neurons_i, neurons_e, w_ie, 0.5,
					zeta, kappa, tau_inh_stdp, wmax, GABA); 
		} else {
			con_ie = new SparseConnection(neurons_i, neurons_e, w_ie, 0.5, GABA);
		}

		SparseConnection * con_ii = new SparseConnection(neurons_i, neurons_i, w_ii, sparseness, GABA);
		con_ii->write_to_file(sys->fn("con_ii","wmat")); // store for reference
		// con_ie = new SparseConnection(neurons_i, neurons_e, w_ie, sparseness, GABA);

		BinarySpikeMonitor * smon_i = new BinarySpikeMonitor( neurons_i, sys->fn("inh","spk"));
		PopulationRateMonitor * pmon_i = new PopulationRateMonitor( neurons_i, sys->fn("inh","prate"), 100 );
	}


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
		if ( ni != 0 ) {
			con_ie->write_to_file(sys->fn("con_ie","wmat"));
		}
	}

	if (errcode) {
		auryn_abort(errcode);
	}

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;
}
