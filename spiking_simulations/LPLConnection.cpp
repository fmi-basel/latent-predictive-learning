/* 
* Copyright 2014-2018 Friedemann Zenke
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

#include "LPLConnection.h"

using namespace auryn;

AurynFloat LPLConnection::beta = 1.0/1e-3; //!< steepness/temperature of nonlinearity

void LPLConnection::init(AurynFloat eta, AurynFloat tau_mean, AurynFloat tau_sigma2, AurynFloat maxweight, AurynState tau_syn, AurynState tau_mem)
{
	if ( !dst->evolve_locally() ) return;
	logger->debug("LPLConnection init");

	// // we check this and throw an error if the user tries to run on more than one rank
	// if ( sys->get_com()->size() > 1 ) {
	// 	logger->msg("LPLConnection can currently not be run in parallel. Aborting.", ERROR);
	// 	throw AurynGenericException();
	// }

	eta_ = eta;
	auryn::logger->parameter("eta",eta);

	delta = 1e-5; // strength of transmitter triggered plasticity
	auryn::logger->parameter("delta",delta);

	epsilon = 1e-32; //!< for division by zero cases

	approximate = false; //!< when enabled small error signals are not back-propagated 
	gamma = 1e-7;       //!< quasi zero norm for error signal ( this value should be set to float precision )

	tau_syn_ = tau_syn;
	tau_mem_ = tau_mem;

	tau_el_rise = 2e-3;
	tau_el_decay = 10e-3;

	tau_vrd_rise = tau_el_rise;
	tau_vrd_decay = tau_el_decay;

	partial_enabled = true;
	augment_gradient = true;
	use_layer_lr = false; //!< set to false for parameter-wise learning rate

	set_min_weight(0.0);
	set_max_weight(maxweight);

	plasticity_sign = 1.0;


	err = dst->get_state_vector("err");

	// traces for van Rossum Distance
	tr_err = dst->get_post_state_trace(err, tau_vrd_rise);
	tr_err_flt = dst->get_post_state_trace(tr_err, tau_vrd_decay);

	tau_avg_err = 10.0;
	// eta_avg_err = auryn_timestep/tau_avg_err;
	mul_avgsqrerr = std::exp(-auryn_timestep/tau_avg_err);
	avgsqrerr = dst->get_state_vector("avgsqrerr");
	temp = dst->get_state_vector("_temp");

	// normalization factor for avgsqrerr
	const double a = tau_vrd_decay;
	const double b = tau_vrd_rise;
	scale_tr_err_flt = 1.0/(std::pow((a*b)/(a-b),2)*(a/2+b/2-2*(a*b)/(a+b)))/tau_avg_err;
    // std::cout << scale_tr_err_flt << std::endl;


	// pre trace 
	tr_pre     = src->get_pre_trace(tau_syn_); // new EulerTrace( src->get_pre_size(), tau_syn );
	tr_pre_psp = new EulerTrace( src->get_pre_size(), tau_mem_ );
	tr_pre_psp->set_target(tr_pre);

	tr_post_sigma2  = new EulerTrace( dst->get_post_size(), tau_sigma2 );


	// tr_post_mean = new EulerTrace( dst->get_post_size(), tau_mean);
	tr_post = dst->get_post_trace(100e-3); // FIXME make adjustable
	tr_post_mean = dst->get_post_trace(tau_mean);

	stdp_active = true; //!< Only blocks weight updates if disabled
	plasticity_stack_enabled = true;

	// sets timecourse for update dynamics
	// timestep_rmsprop_updates = 5000 + 257;
	timestep_rmsprop_updates = 5000;
	set_tau_rms(300.0);
	logger->parameter("timestep_rmsprop_updates", (int)timestep_rmsprop_updates);


	set_tau_weight_decay(1800); // set weight decay to default value 30mins
	use_weight_decay = false;

	// compute delay size in AurynTime
	const double finite_difference_time_delta = 0e-3; // we just keep this in case we still need it at some point
	delay_size = finite_difference_time_delta/auryn_timestep;
	logger->parameter("delay_size", delay_size);

	
	double post_delay_size = 20e-3; // TODO make adjustable 
	post_delay = new SpikeDelay( );
	post_delay->set_delay(post_delay_size/auryn_timestep); 
	post_delay->set_clock_ptr(sys->get_clock_ptr());


	partial_delay = new AurynDelayVector( dst->get_post_size(), delay_size+MINDELAY );
	pre_psp_delay = new AurynDelayVector( src->get_pre_size(), delay_size+MINDELAY ); 


	logger->debug("LPLConnection complex matrix init");
	// Set number of synaptic states
	w->set_num_synapse_states(5);

	zid_weight = 0;
	w_val   = w->get_state_vector(zid_weight);

	zid_el = 1;
	el_val = w->get_state_vector(zid_el);

	zid_el_flt = 2;
	el_val_flt = w->get_state_vector(zid_el_flt);

	zid_sum = 3;
	el_sum  = w->get_state_vector(zid_sum);

	zid_grad2 = 4;
	w_grad2 = w->get_state_vector(zid_grad2);
	// w_grad2->set_all(1.0e-3);

	// Run finalize again to rebuild backward matrix
	logger->debug("LPLConnection complex matrix finalize");
	DuplexConnection::finalize();

	// store instance in static set
	// m_instances.insert(this);
}


void LPLConnection::free()
{
	logger->debug("LPLConnection free");
	// store instance in static set
	// m_instances.erase(this);

	delete tr_pre_psp;
	delete tr_post_sigma2;
	delete partial_delay;
	delete pre_psp_delay;
}

LPLConnection::LPLConnection(
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat lambda, 
		AurynFloat phi, 
		AurynFloat tau_mean, 
		AurynFloat tau_sigma2, 
		AurynFloat maxweight, 
		TransmitterType transmitter,
		std::string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(eta, tau_mean, tau_sigma2, maxweight);
	lambda_ = lambda;
	phi_ = phi;
	if ( name.empty() )
		set_name("LPLConnection");
}

LPLConnection::~LPLConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

AurynWeight LPLConnection::instantaneous_partial(NeuronID loc_post)
{
	if ( !partial_enabled ) return 1.0;

	// compute pseudo partial
	// const AurynFloat h = (dst->mem->get(loc_post)+50e-3)*beta;
	const AurynState voltage = dst->mem->get(loc_post);
	if ( voltage < -80e-3 ) return 0.0;
	const AurynFloat h = (voltage+50e-3)*beta;
	const AurynFloat part = beta/std::pow((1.0+std::abs(h)),2);
	return part;
}

void LPLConnection::add_to_syntrace(const AurynLong didx, const AurynDouble input)
{
	// computes plasticity update
	AurynWeight * elt = el_val->ptr(didx);
	*elt += input; 
}

void LPLConnection::add_to_err(NeuronID spk, AurynState val) 
{
	if ( dst->localrank(spk) ) {
		const NeuronID trspk = dst->global2rank(spk);
		err->add_specific(trspk, val);
	}
}


/* \brief Computes error signal */
void LPLConnection::compute_err()
{
	SpikeContainer * sc = dst->get_spikes();
	SpikeContainer * sd = post_delay->get_spikes();

	err->set_zero(); // reset error state

	// compute the Hebb loss
	err->copy(tr_post_mean);
	// err->add(0.5); // FIXME remove -- hint on two timescale moving threshold? 
	err->scale(-1.0*auryn_timestep/tr_post_mean->get_tau()); // normalize by mem scale
	// add spikes
	for ( unsigned int i = 0 ; i < sc->size(); ++i ) {
		const NeuronID trspk = dst->global2rank(sc->at(i));
		// float trmod = tr_post->get(trspk);
		err->add_specific(trspk, 1.0);
	}
	temp->copy(tr_post_sigma2);
	temp->add(1e-3); // TODO make a parameter called xi to be used denominator
	err->div(temp);
	err->scale(lambda_);
	err->add(delta); // add transmitter triggered plasticity 

	// Compute -dzdt error signal on spike trains using the van Rossum trick
	for ( unsigned int i = 0 ; i < sc->size(); ++i ) add_to_err(sc->at(i), -1.0*phi_);
	for ( unsigned int i = 0 ; i < sd->size(); ++i ) add_to_err(sd->at(i), 1.0*phi_); // these are the targets

	// some qnd monitoring
	if ( false && sys->get_clock()%100000==0) std::cout << std::scientific 
		<< " avgsqrerr " << avgsqrerr->mean()
		<< ", mean firing rate " << tr_post_mean->mean()/tr_post_mean->get_tau() 
		<< ", tr_post_sigma2 " << tr_post_sigma2->mean() 
		<< std::endl;


	// TESTING
	// for ( NeuronID i = 0 ; i < dst->get_rank_size(); ++i ) {
	// 	if ( dst->mem->get(i) > -51e-3 ) add_to_err(i, -1e-3);
	// }
}

/*! \brief Computes local part of plasticity rule without multiplying the error signal
 *
 * Computes: epsilon*( sigma_prime(u_i) PSP_j )
 * */
void LPLConnection::process_plasticity()
{
	// compute partial deriviatives and store in delay
	for ( NeuronID i = 0 ; i < dst->get_post_size() ; ++i )
		partial_delay->set( i, instantaneous_partial(i) );

	// compute psp and store in delay
	for ( NeuronID j = 0 ; j < src->get_pre_size() ; ++j )
		pre_psp_delay->set( j, tr_pre_psp->get(j) );

	// loop over all pre neurons
	for (NeuronID j = 0; j < src->get_pre_size() ; ++j ) {
		const AurynState psp = pre_psp_delay->mem_get(j);
		if ( approximate && psp <= gamma ) { continue; } 
		// std::cout << std::scientific << psp << std::endl;
		
		// loop over all postsynaptic partners
	    for (const NeuronID * c = w->get_row_begin(j) ; 
					   c != w->get_row_end(j) ; 
					   ++c ) { // c = post index
			// compute data index for address in complex array
			const NeuronID li = dst->global2rank(*c);
			const AurynState sigma_prime = partial_delay->mem_get(li); 
			const AurynLong didx   = w->ind_ptr_to_didx(c); 

			// compute eligibility trace
			const AurynWeight syn_trace_input = plasticity_sign*psp*sigma_prime; 
			add_to_syntrace( didx, syn_trace_input );
		}
	}

	partial_delay->advance(); // now 'get' points to the delayed version
	pre_psp_delay->advance(); // now 'get' points to the delayed version

	// # SECOND compute outer convolution of synaptic traces
	const AurynFloat mul_follow = auryn_timestep/tau_el_decay;
	el_val_flt->follow(el_val, mul_follow);

	const AurynFloat scale_const = std::exp(-auryn_timestep/tau_el_rise);
	el_val->scale(scale_const);

	// # THIRD compute correlation between el_val_flt and the filtered error signal
	// and store in el_sum 'the summed eligibilty trace'

	for (NeuronID li = 0; li < dst->get_post_size() ; ++li ) {
		if ( approximate && std::abs(tr_err_flt->get(li)) <= gamma ) { continue; }
		const NeuronID gi = dst->rank2global(li); 
		for (const NeuronID * c = bkw->get_row_begin(gi) ; 
				c != bkw->get_row_end(gi) ; 
				++c ) {
			AurynWeight * weight = bkw->get_data(c); 
			const AurynLong didx = w->data_ptr_to_didx(weight); 
			const AurynState e = tr_err_flt->get(li);
			AurynWeight de = el_val_flt->get(didx)*e; 
			el_sum->add_specific(didx, de);
		}
	}
}


void LPLConnection::propagate_forward()
{

   // loop over all spikes
   for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
				   spike != src->get_spikes()->end() ; ++spike ) {
	   // loop over all postsynaptic partners
	   for ( AurynLong c = w->get_row_begin_index(*spike) ;
					   c != w->get_row_end_index(*spike) ; 
					   ++c ) { // c = post index

			   // transmit signal to target at postsynaptic neuron
			   // AurynWeight * weight = w->get_data_ptr(c); 
			   // transmit( *c , *weight );
			   transmit( w->get_colind(c) , w->get_value(c) );
	   }
   }
}


template <typename T> int sgn(T val) {
	    return (T(0) < val) - (val < T(0));
}



void LPLConnection::propagate()
{
	// propagate spikes
	propagate_forward();
}

template <typename T>
T clipn(const T& n, const T& lower, const T& upper) {
	  return std::max(lower, std::min(n, upper));
}

void LPLConnection::evolve()
{
	if ( !plasticity_stack_enabled || !dst->evolve_locally()) return;

	post_delay->get_spikes_immediate()->clear();
	post_delay->push_back(dst->get_spikes_immediate());

	// compute squared error vector of this layer
	temp->copy(tr_err_flt);
	// temp->mul(scale_tr_err_flt);
	temp->sqr();
	temp->scale(auryn_timestep);
	avgsqrerr->scale(mul_avgsqrerr);
	avgsqrerr->add(temp);

	// compute the LPL error
	compute_err();


	// add nonlinear Hebb (pre post correlations) to synaptic traces and filter these traces
	process_plasticity();

	if ( auryn::sys->get_clock()%timestep_rmsprop_updates == 0  ) {
	// std::cout << std::scientific << "  tr_err_flt=" << tr_err_flt->mean() 
	//           << std::scientific << "  avgsqrerr=" << avgsqrerr->mean() << std::endl;

		double gm = 0.0;
		if ( augment_gradient ) {
			// evolve complex synapse parameters
			for ( AurynLong k = 0; k < w->get_nonzero(); ++k ) {
				// AurynWeight * weight = w->get_data_ptr(k, zid_weight);
				AurynWeight * minibatch = w->get_data_ptr(k, zid_sum);
				AurynWeight * g2 = w->get_data_ptr(k, zid_grad2);

				// copies our low-pass value "minibatch" to grad 
				const AurynFloat grad = (*minibatch)/timestep_rmsprop_updates;
				// *minibatch = 0.0f;

				// update moving averages 
				*g2 = std::max( grad*grad, rms_mul* *g2 );

				// To implement RMSprop we  need this line
				// *g2 = rms_mul* *g2 + (1.0-rms_mul)*std::pow(grad,2) ;
			}

			if ( use_layer_lr ) {
				gm =w->get_synaptic_state_vector(zid_grad2)->max();
			}
		}



		for ( AurynLong k = 0; k < w->get_nonzero(); ++k ) {
			AurynWeight * weight = w->get_data_ptr(k, zid_weight);
			AurynWeight * minibatch = w->get_data_ptr(k, zid_sum);

			// copies our low-pass value "minibatch" to grad and erases it
			const AurynDouble grad = (*minibatch)/timestep_rmsprop_updates;
			*minibatch = 0.0f; // reset mini batch

			// carry out weight updates
			if ( stdp_active ) {

				// dynamic gradient rescaling
				// (per parameter learning rate)
				if ( !use_layer_lr ) {
					gm = w->get_synaptic_state_vector(zid_grad2)->get(k);
				}

		
				double rms_scale = 1.0;
				if ( augment_gradient ) rms_scale = 1.0/(std::sqrt(gm)+epsilon);
		
				// update weight
				if ( use_weight_decay ) *weight += -scl_weight_decay * *weight * eta_; // weight decay
				*weight += rms_scale * grad * eta_;  // surrogate gradient based update

				// clip weight
				if ( *weight < get_min_weight() ) *weight = get_min_weight();
				else if ( *weight > get_max_weight() ) *weight = get_max_weight();
			}
		}

		// DEBUG
		// AurynSynStateVector * tmp = w->get_synaptic_state_vector(zid_el_flt);
		// tmp->zero_effective_zeros(1.0);
		// std::cout << "mean=" << tmp->mean() << std::endl;
		// std::cout << "nz=" << 1.0*tmp->nonzero()/tmp->size << std::endl;

	}


	
	// update follow traces which are not registered with the kernel
	tr_pre_psp->follow();
	// tr_post_mean->follow();

	// compute moving sigma2
	// subtract mean
	temp->copy(tr_post_mean);
	temp->scale(-1.0*auryn_timestep/tr_post_mean->get_tau()); // normalize by mem scale
	// add spikes
	SpikeContainer * sc = dst->get_spikes();
	for ( unsigned int i = 0 ; i < sc->size(); ++i ) {
		const NeuronID trspk = dst->global2rank(sc->at(i));
		temp->add_specific(trspk, 1.0);
	}
	// compute square
	temp->sqr();

	// add to moving tr_post_sigma2
	temp->scale(1.0/tr_post_sigma2->get_tau()); // correct for mean time scale 
	tr_post_sigma2->add(temp);
	tr_post_sigma2->evolve();
}

void LPLConnection::set_tau_rms( AurynState tau )
{
	tau_rms = tau;
	rms_mul = std::exp(-auryn_timestep*timestep_rmsprop_updates/tau_rms);
	auryn::logger->parameter("rms_mul",rms_mul);
}

void LPLConnection::set_tau_weight_decay( AurynState tau )
{
	tau_weight_decay = tau;
	scl_weight_decay = 1.0-std::exp(-auryn_timestep*timestep_rmsprop_updates/tau_weight_decay); // weight decay strength 
	auryn::logger->parameter("scl_weight_decay",scl_weight_decay);
	use_weight_decay = true;
}

void LPLConnection::set_tau_syn( AurynState tau )
{
	tau_syn_ = tau;
	tr_pre->set_timeconstant(tau_syn_); 
}

void LPLConnection::set_tau_mem( AurynState tau )
{
	tau_mem_ = tau;
	tr_pre_psp->set_timeconstant(tau_mem_); 
}


double LPLConnection::get_mean_square_error( )
{
	return scale_tr_err_flt*avgsqrerr->mean()/(1.0-std::exp(-sys->get_time()/tau_avg_err)+1e-9);
}

void LPLConnection::set_learning_rate( double eta )
{
	eta_   = eta;
}



