#ifndef    HMATRIX_H
#define    HMATRIX_H

#include <cstdlib>
#include <cstdio>
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include "common.h"
#include "basis.h" 

#define PI 3.14159265358979323846
//constexpr auto PI = 3.14159265358979323846;

class Parameters
{
public:
	// Basis
	int LatticeSize, NoParticles;
	// choose a model, do it later
	void SetDefault();
	
	// 
	char calc_eigvec;
	// static observables 
	char eig_IPR;		// inverse participation ratio: sum |phi_a|^4
	char eig_SE;		// shannon entropy:				- \sum |phi_a|^2 ln (|phi_a|^2)
//	char eig_PE;		// participation entropy:		- ln (IPR)
	char eig_EE;
	char eig_ExtremeP;
	// dynamic
	char prt_Csq_p_alpha;		// print |<p|alpha>|^2, very disk consuming
	char prt_IG_p_aa;
	char prt_IGsq_p_aa;
	// diagonal essemble results
	char de_IG;
	char de_IGsq;
	char de_mz;
	//char de_ExtremeP;

	// number of initial states for a certain energy density 
	int N_p_de;
	int N_p_evo;
    // target energy density, and energy window width
    double target_epsilon;
    double ini_epsi_width;

	// evolution
//	double evo_time_min;
	double evo_time_max;
	int evo_time_steps;
	int evo_time_IncStyle;	// 0/1 for linear/exponetial time increase
	double *time_vec;
	void GetTimeVec();

	char evo_IPR;
	char evo_SE;
	char evo_EE;
	char evo_IG;
	char evo_IGsq;
	char evo_mzit;
	//char evo_ExtremeP;

	// disorder
	double Rand_V;
	// stark localization parameters 
	double Stark_gamma;
	double Stark_alpha;
	double* Rand_Vec;
	// 
	void Read_input();

	// specific model parameters
	// XXZ model 
	double XXZ_g;
	// SC Qubits
	void Read_input_SCQubits();
	double* Qubits_Jij;
	char Qubits_cut_longrange;
	int Qubits_cut_to_n;
};

class Dyn_DataStruc
{
public:

	void Initialize(Parameters& para, const l_int& _dim, const int& _prt_ind, const double& _epsilon);
	void ReleaseSpace(Parameters& para);
	void PrintDynResults(Parameters& para);

	double varepsilon;
	int prt_ind;

	int dim;

	double* Csq_p_alpha;
	double* IG_p_q;
	double* IG_p_aa;
	double* IGsq_p_q;
	double* IGsq_p_aa;

	double IG_DE_p;
	double IGsq_DE_p;

	//int t_len;
	//double* time_vec;

	double* IG_p_t;
	double* IGsq_p_t;
	double* IPR_p_t;
	double* SE_p_t;
	double* EE_p_t; 
	//double* ExtremeP_p_t;
	double* mzi_p_t;

	double* wf0_wft_inner;
};

class Hmatrix
{
public:
	Hmatrix();
	Parameters Params;
	~Hmatrix();

	// Build Hamiltonian matrix for 
	// XXZ model
	void Build_HamMat_XXZ(Basis& basis);
	// general XXZ, with site-dependent couplings 
	void Build_HamMat_XXZ_general(Basis& basis);

    // SC Qubits
	void Build_HamMat_SCQubits(Basis& basis);
	void Calc_CoeffMat_SCQubits();
	// add more later

	// ===========================================
	// General lattice information
	l_int Dim;
	int LatticeSize;
	int NoParticles;
	double* Jij;    // hopping matrix for models with long-range couplings
	double* hii;    // on-site disorder

	// dense matrix form 
	double* Hmat;   // Hamiltonian matrix
	double* H_diag; // digonal part of Hs, energy of Fock state
	double* spec;   // eigenvalues
    void Check_Mat_Conj(double *mat, l_int dim);
	// sparse matrix storage for later use, like H|psi>
	lapack_int nnz;
	double* SMat_vals;
	lapack_int* SMat_cols;
	lapack_int* SMat_PointerBE;
	// dense to sparse (csr) represetation for symmetric matrix 
	void Matrix_dense2csr_upper();
	void Calc_Sparse_Hsquare(const l_int& p, const double &epsilon, const int&prt_ind);

	// static quantities 
	void Calc_Static_Quantities(Basis& basis);
	void Static_roag();

	void H_fetch_eigvec(const l_int& k, double* wf);
	void Static_IPR();
	void Static_SE();
	void Static_ExtremeP(Basis& basis);
	double Get_ExtremeP_from_mzVec(double* _mzi);
	void Get_ni_wf(Basis& basis, double* _wf, double* _ni);
	void Get_mzi_wf(Basis& basis, double* _wf, double* _ni);
	void Get_ni_wfsq(Basis& basis, double* _wfsq, double* _ni);
	void Get_mzi_wfsq(Basis& basis, double* _wfsq, double* _ni);

	//
	//l_int Get_TargetFock(const double& Target_E, std::vector <std::pair<double, l_int> >& Fock_E_n);
	l_int Get_TargetFock_left(const double& Target_E, std::vector <std::pair<double, l_int> >& Fock_E_n);
	l_int Get_TargetFock_right(const double& Target_E, std::vector <std::pair<double, l_int> >& Fock_E_n);

	// ---------------------------------------------------
	// Compute useful vectors of the initial Fock state p
	// p,q: index in configuration basis; 
	// alpha,beta: index in eigen basis
	// ---------------------------------------------------
	// \sum_q <q|I(p)|q> 
	void Get_IG_p_q(Basis& basis, const l_int& p, double* IG_p_q);
	// \sum_q <q|I^2(p)|q>
	void Get_IGsq_p_q(Basis& basis, const l_int& p, double* IGsq_p_q);
	// \sum_alpha |<alpha|p>|^2
	void Get_Csq_p_alpha(const l_int& p, double* Csq_p_alpha);
	// \sum_alpha <alpha|O(p)|alpha>
	void Get_O_p_aa(double* O_p_q, double* O_p_aa);

	double Get_O_p_DE(double* IG_p_q, double* C_alpha2_p, double* IG_p_alpha);
	double Get_Width_p(const double& E_real, double* C_alpha2_p);

	// dynamic quantities
	void Calc_Dynamic_Quantities(Basis& basis);

	// real time evolution
	void Calc_evolution_p(Basis& basis, const int& p, Dyn_DataStruc& dyn_data);

	double Compute_IPR_wfsq(const double* wfsq);
	double Compute_Shannon_Entropy_wfsq(const double* wfsq);

	// get time depedent wave function, then compute required observables 

/*
	// outputs -----------------------------------------------------------------------------
	// roag
	double* roag;
	void H_Spec_Roag();		    // Eigen values, and ratio of adjecent gaps,  do not need eigen vectorss
	// IPR, shannon entropy
	double* IPR;
	double* SE;
	void H_IPR_SE(Basis& basis);
	// entanglement entropy
	double* ee;
	double Calc_entanglement_entropy(Basis& basis, double* wf_in, const int& size_A, const int& size_B);
	void H_Entanglement_Entropy(Basis& basis);      // Entanglement entropy
	// charge on side of the chain
	double* N_oneside;
	void Get_totN_p_FockBasis(Basis& basis, const l_int& num_p, double* N_p);
	void H_Cal_OneSideCharge(Basis& basis);
*/
// debug
	inline void PrintHMatrix(int aindex);
};

#endif
