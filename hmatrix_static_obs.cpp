#include "hmatrix.h"
#include <omp.h>

using namespace std;

void Hmatrix::Calc_Static_Quantities(Basis& basis)
{
	Static_roag();

	if ('y' == Params.eig_IPR)
		Static_IPR();

	if ('y' == Params.eig_SE)
		Static_SE();

	if ('y' == Params.eig_ExtremeP)
		Static_ExtremeP(basis);
}

void Hmatrix::Static_roag()
{
	double* gaps = new double[Dim - 1];
	for (unsigned long long i = 0; i < Dim - 1; i++) gaps[i] = spec[i + 1] - spec[i];
	double* roag = new double[Dim - 2];
	for (l_int i = 0; i < Dim - 2; i++) {
		if (0 == gaps[i] && 0 == gaps[i + 1]) roag[i] = 1;
		else roag[i] = (gaps[i] < gaps[i + 1]) ? gaps[i] / gaps[i + 1] : gaps[i + 1] / gaps[i];
	}
	Vec_fwrite_double("roag.bin", roag, Dim-2);
	delete[]gaps;
	delete[]roag;
}

void Hmatrix::H_fetch_eigvec(const l_int& k, double* wf) 
{
	vdPackI(Dim, &Hmat[k], Dim, wf);
}

void Hmatrix::Static_IPR()
{
	double* IPR = new double[Dim];
//#pragma omp parallel for schedule(dynamic)
	for (l_int i = 0; i < Dim; i++) {
		double* wf = new double[Dim];
		H_fetch_eigvec(i, wf);
		vdSqr(Dim, wf, wf);
		IPR[i] = Compute_IPR_wfsq(wf);
		delete[]wf;
	}
	Vec_fwrite_double("IPR.bin", IPR, Dim);
	delete[] IPR;
}

void Hmatrix::Static_SE()
{
	double* SE = new double[Dim];
	//#pragma omp parallel for schedule(dynamic)
	for (l_int i = 0; i < Dim; i++) {
		double* wf = new double[Dim];
		H_fetch_eigvec(i, wf);
		vdSqr(Dim, wf, wf);
		SE[i] = Compute_Shannon_Entropy_wfsq(wf);
		delete[]wf;
	}
	Vec_fwrite_double("SE.bin", SE, Dim);
	delete[] SE;
}


void Hmatrix::Static_ExtremeP(Basis& basis)
{
	double* mz_ik = new double[LatticeSize * Dim];
	double* ExtremeP = new double[Dim];
	for (int k = 0; k < Dim; k++)
	{
		double* wf = new double[Dim];
		H_fetch_eigvec(k, wf);
		Get_mzi_wf(basis, wf, &mz_ik[k * LatticeSize]);
		ExtremeP[k] = Get_ExtremeP_from_mzVec(&mz_ik[k * LatticeSize]);
		delete[] wf;
	}
	Vec_fwrite_double("ExtremeP.bin", ExtremeP, Dim);
	Vec_fwrite_double("mz_ik.bin", mz_ik, LatticeSize * Dim);
	// 
	delete[] mz_ik;
	delete[] ExtremeP;
}

void Hmatrix::Get_ni_wf(Basis& basis, double* _wf, double* _ni)
{
	for (int i = 0; i < LatticeSize; i++) _ni[i] = 0;
	for (int ik = 0; ik < Dim; ik++)
	{
		int Num_k = basis.get_state(ik);
		for (int i = 0; i < LatticeSize; i++)
		{
			_ni[i] += ((Num_k >> i) & 1) * _wf[ik] * _wf[ik];
		}
	}
}

void Hmatrix::Get_ni_wfsq(Basis& basis, double* _wfsq, double* _ni)
{
	for (int i = 0; i < LatticeSize; i++) _ni[i] = 0;
	for (int ik = 0; ik < Dim; ik++)
	{
		int Num_k = basis.get_state(ik);
		for (int i = 0; i < LatticeSize; i++)
		{
			_ni[i] += ((Num_k >> i) & 1) * _wfsq[ik];
		}
	}
}

void Hmatrix::Get_mzi_wf(Basis& basis, double* _wf, double* _ni)
{
	for (int i = 0; i < LatticeSize; i++) _ni[i] = 0;
	for (int ik = 0; ik < Dim; ik++)
	{
		int Num_k = basis.get_state(ik);
		for (int i = 0; i < LatticeSize; i++)
		{
			_ni[i] += (((Num_k >> i) & 1) - 0.5) * _wf[ik] * _wf[ik];
		}
	}
}

void Hmatrix::Get_mzi_wfsq(Basis& basis, double* _wfsq, double* _ni)
{
	for (int i = 0; i < LatticeSize; i++) _ni[i] = 0;
	for (int ik = 0; ik < Dim; ik++)
	{
		int Num_k = basis.get_state(ik);
		for (int i = 0; i < LatticeSize; i++)
		{
			_ni[i] += (((Num_k >> i) & 1) - 0.5) * _wfsq[ik];
		}
	}
}

double Hmatrix::Get_ExtremeP_from_mzVec(double* _mzi)
{
	double* mz_abs = new double[LatticeSize];
	for (int i = 0; i < LatticeSize; i++)
		mz_abs[i] = std::abs(_mzi[i]);
	double extremeP = 0.5 - *std::max_element(mz_abs, mz_abs + LatticeSize);
	delete[] mz_abs;
	return extremeP;
}