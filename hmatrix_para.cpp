#include "hmatrix.h"
using namespace std;
#include <omp.h>

// initialize default values
void Parameters::SetDefault()
{
	eig_IPR = 'n';
	eig_SE = 'n';
	//	char eig_PE = 'n';
	eig_EE = 'n';
	eig_ExtremeP = 'n';
	// dynamic
	prt_Csq_p_alpha = 'n';
	prt_IG_p_aa = 'n';
	prt_IGsq_p_aa = 'n';
	// diagonal essemble results
	de_IG = 'n';
	de_IGsq = 'n';
	//de_ExtremeP = 'n';

	// number of initial states for a certain energy density 
	N_p_de = 1;
	N_p_evo = 1;

	// evolution
//	double evo_time_min = 1;
	evo_time_max = 100;
	evo_time_steps = 100;
	evo_time_IncStyle = 0;	// 0/1 for linear/exponetial time increase

	evo_IPR = 'n';
	evo_SE = 'n';
	evo_EE = 'n';
	evo_IG = 'n';
	evo_IGsq = 'n';
	//evo_ExtremeP = 'n';

	// disorder
	Rand_V = 0;
	// stark localization parameters 
	Stark_gamma = 0;
	Stark_alpha = 0;

	// specific model parameters
	// XXZ model 
	XXZ_g = 1.0;
	// SC Qubits
	Qubits_cut_longrange = 'n';
	Qubits_cut_to_n = 1;
}

void Parameters::Read_input()
{
	GetParaFromInput_int("input.in", "LatticeSize", LatticeSize);
	GetParaFromInput_int("input.in", "NoParticles", NoParticles);
	//
	GetParaFromInput_real("input.in", "Rand_V", Rand_V);
	GetParaFromInput_real("input.in", "Stark_alpha", Stark_alpha);
	GetParaFromInput_real("input.in", "Stark_gamma", Stark_gamma);
	//
	GetParaFromInput_char("input.in", "eig_IPR", eig_IPR);
	GetParaFromInput_char("input.in", "eig_SE", eig_SE);
	GetParaFromInput_char("input.in", "eig_EE", eig_EE);
	GetParaFromInput_char("input.in", "eig_ExtremeP", eig_ExtremeP);
	//
	GetParaFromInput_char("input.in", "prt_Csq_p_alpha", prt_Csq_p_alpha);
	GetParaFromInput_char("input.in", "prt_IG_p_aa", prt_IG_p_aa);
	GetParaFromInput_char("input.in", "prt_IGsq_p_aa", prt_IGsq_p_aa);
	//
    GetParaFromInput_real("input.in", "target_epsilon", target_epsilon);
	GetParaFromInput_real("input.in", "ini_epsi_width", ini_epsi_width);
	GetParaFromInput_int("input.in", "N_p_de", N_p_de);
	GetParaFromInput_char("input.in", "de_IG", de_IG);
	GetParaFromInput_char("input.in", "de_IGsq", de_IGsq);
	//GetParaFromInput_char("input.in", "de_ExtremeP", de_ExtremeP);
	//
	GetParaFromInput_int("input.in", "N_p_evo", N_p_evo);
	//	GetParaFromInput_real("input.in", "evo_time_min", evo_time_min);
	GetParaFromInput_real("input.in", "evo_time_max", evo_time_max);
	GetParaFromInput_int("input.in", "evo_time_steps", evo_time_steps);
	GetParaFromInput_int("input.in", "evo_time_IncStyles", evo_time_IncStyle);

	GetParaFromInput_char("input.in", "evo_IG", evo_IG);
	GetParaFromInput_char("input.in", "evo_IGsq", evo_IGsq);
	GetParaFromInput_char("input.in", "evo_IPR", evo_IPR);
	GetParaFromInput_char("input.in", "evo_SE", evo_SE);
	GetParaFromInput_char("input.in", "evo_EE", evo_EE);
	//GetParaFromInput_char("input.in", "evo_ExtremeP", evo_ExtremeP);

	//
	if ('y' == eig_IPR || 'y' == eig_ExtremeP || 'y' == eig_EE || 'y' == de_IG || 'y' == de_IGsq || 'y' == evo_IPR || 'y' == evo_IG || 'y' == evo_IGsq || 'y' == evo_EE)
		calc_eigvec = 'y';
	cout << endl << "calc_eigvec = " << calc_eigvec << endl << endl;

	//
	GetTimeVec();
}

void::Parameters::GetTimeVec()
{
	int nt = evo_time_steps;
	time_vec = new double[nt];
	// linear time increase
	if (0 == evo_time_IncStyle)
	{
		for (int i = 0; i < nt; i++)
		{
			time_vec[i] = i * evo_time_max / nt;
		}
	}

	else if (1 == evo_time_IncStyle)
	{
		double tmaxlog = log10(evo_time_max);
		double tminlog = log10(1e-1);
		time_vec[0] = 0;
		for (int i = 1; i < nt; i++)
		{
			time_vec[i] = pow(10, tminlog + i * (tmaxlog - tminlog) / nt);
		}
	}
}

void Dyn_DataStruc::Initialize(Parameters& para, const l_int& _dim, const int& _prt_ind, const double& _epsilon)
{
	dim = _dim;
	prt_ind = _prt_ind;
	varepsilon = _epsilon;

	// necessaraties for DE & evo
	if ('y' == para.de_IG || 'y' == para.evo_IG)
		IG_p_q = new double[dim];
	if ('y' == para.de_IGsq || 'y' == para.evo_IGsq)
		IGsq_p_q = new double[dim];

	// necessaraties for DE
	if ('y' == para.de_IG || 'y' == para.prt_IG_p_aa || 'y' == para.de_IGsq || 'y' == para.prt_IGsq_p_aa)
		Csq_p_alpha = new double[dim];
	if ('y' == para.de_IG || 'y' == para.prt_IG_p_aa)
		IG_p_aa = new double[dim];
	if ('y' == para.de_IGsq || 'y' == para.prt_IGsq_p_aa)
		IGsq_p_aa = new double[dim];

	// necessaraties for evo
	int t_len = para.evo_time_steps;
	//GetTimeVec(para);

	//  
	wf0_wft_inner = new double[t_len];
	if ('y' == para.evo_IPR)
		IPR_p_t = new double[t_len];
	if ('y' == para.evo_SE)
		SE_p_t = new double[t_len];
	if ('y' == para.evo_IG)
		IG_p_t = new double[t_len];
	if ('y' == para.evo_IGsq)
		IGsq_p_t = new double[t_len];
	if ('y' == para.evo_EE)
		EE_p_t = new double[t_len];
	//if ('y' == para.evo_ExtremeP)
	if ('y' == para.evo_mzit)
	{
//		ExtremeP_p_t = new double[t_len];
		mzi_p_t = new double[t_len * para.LatticeSize];
	}

}

void Dyn_DataStruc::ReleaseSpace(Parameters& para)
{
	// necessaraties for DE & evo
	if ('y' == para.de_IG || 'y' == para.evo_IG)
		delete[]IG_p_q;
	if ('y' == para.de_IGsq || 'y' == para.evo_IGsq)
		delete[]IGsq_p_q;

	// necessaraties for DE
	if ('y' == para.de_IG || 'y' == para.prt_IG_p_aa || 'y' == para.de_IGsq || 'y' == para.prt_IGsq_p_aa)
		delete[]Csq_p_alpha;
	if ('y' == para.de_IG || 'y' == para.prt_IG_p_aa)
		delete[]IG_p_aa;
	if ('y' == para.de_IGsq || 'y' == para.prt_IGsq_p_aa)
		delete[]IGsq_p_aa;

	// necessaraties for evo
	delete[]wf0_wft_inner;
	if ('y' == para.evo_IPR)
		delete[]IPR_p_t;
	if ('y' == para.evo_SE)
		delete[]SE_p_t;
	if ('y' == para.evo_IG)
		delete[]IG_p_t;
	if ('y' == para.evo_IGsq)
		delete[]IGsq_p_t;
	if ('y' == para.evo_EE)
		delete[]EE_p_t;
	//if ('y' == para.evo_ExtremeP) 
	if ('y' == para.evo_mzit) 
	{
	//	delete[]ExtremeP_p_t;
		delete[]mzi_p_t;
	}
}

void::Dyn_DataStruc::PrintDynResults(Parameters& para) {

	// |<alpha|p>|^2
	if ('y' == para.prt_Csq_p_alpha) {
		char fname[80];
		sprintf(fname, "Csq_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, Csq_p_alpha, dim);
	}

	// <alpha|IG(p)|alpha>
	if ('y' == para.prt_IG_p_aa)
	{
		char fname[80];
		sprintf(fname, "IG_aa_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, IG_p_aa, dim);
	}

	// <alpha|IGsq(p)|alpha>
	if ('y' == para.prt_IGsq_p_aa) {
		char fname[80];
		sprintf(fname, "IGsq_aa_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, IGsq_p_aa, dim);
	}

	int t_len = para.evo_time_steps;
	// real-time evolution results 
	char fname0[80];
	sprintf(fname0, "evo_wftwf0inner_e%0.2f_ind%d.bin", varepsilon, prt_ind);
	Vec_fwrite_double(fname0, wf0_wft_inner, t_len);

	if ('y' == para.evo_IG)
	{
		char fname[80];
		sprintf(fname, "evo_IG_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, IG_p_t, t_len);
	}

	if ('y' == para.evo_IGsq)
	{
		char fname[80];
		sprintf(fname, "evo_IGsq_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, IGsq_p_t, t_len);
	}

	if ('y' == para.evo_IPR)
	{
		char fname[80];
		sprintf(fname, "evo_IPR_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, IPR_p_t, t_len);
	}

	if ('y' == para.evo_SE)
	{
		char fname[80];
		sprintf(fname, "evo_SE_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname, SE_p_t, t_len);
	}

	//if ('y' == para.evo_ExtremeP)
	if ('y' == para.evo_mzit) 
	{
		//char fname[80];
		//sprintf(fname, "evo_ExtremeP_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		//Vec_fwrite_double(fname, ExtremeP_p_t, t_len);
		//
		char fname1[80];
		sprintf(fname1, "evo_mzi_e%0.2f_ind%d.bin", varepsilon, prt_ind);
		Vec_fwrite_double(fname1, mzi_p_t, para.LatticeSize * t_len);
	}
}
