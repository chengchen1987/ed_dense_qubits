#include <functional> 

#include "hmatrix.h"
using namespace std;
#include <omp.h>

void Hmatrix::Calc_Dynamic_Quantities(Basis& basis)
{
    // Step 1: Get E_fock, and ascend E_Fock by energy
    vector <pair<double, l_int> > Fock_E_n;  // stores E_fock, and index of this Fock state
    for (l_int p = 0; p < Dim; p++)
    {
        pair<double, l_int> aux = make_pair(H_diag[p], p);
        Fock_E_n.push_back(aux);
    }
    std::sort(Fock_E_n.begin(), Fock_E_n.end());
    // cout << "Fock_E_n finished" << endl;
    double E_band = spec[Dim - 1] - spec[0];
    double E_min = spec[0];
    // print Fock E
    double* sorted_Fock_epsi = new double[Dim];
    for (l_int k = 0; k < Dim; k++) {
        sorted_Fock_epsi[k] = (Fock_E_n[k].first - E_min) / E_band;
    }
    Vec_fwrite_double("Fock_epsilons.dat", sorted_Fock_epsi, Dim);
    // target varepsilons 

    // initial states for target varepsilons, choose random initial stats in some energy window 
    double epsilon = Params.target_epsilon;
    // set energy window
    double epsi0 = epsilon - Params.ini_epsi_width;
    l_int p0 = Get_TargetFock_right(epsi0*E_band+E_min, Fock_E_n);
    if (-1 == p0) p0 = 1;
    double epsi1 = epsilon + Params.ini_epsi_width;
    l_int p1 = Get_TargetFock_left(epsi1*E_band+E_min, Fock_E_n);
    if (-1 == p1) p1 = Dim - 1;
    std::cout << "epsi0: " << epsi0 << ", p0: " << p0 << ", espi: "<< (Fock_E_n[p0].first-E_min)/E_band << std::endl;
    std::cout << "epsi1: " << epsi1 << ", p1: " << p1 << ", espi: "<< (Fock_E_n[p1-1].first-E_min)/E_band << std::endl;
    // choose random initial states 
    vector <pair<int, l_int> > target_ind_Fock;
    for (int i = 0; i < Params.N_p_de; i++)
    {   
        std::mt19937 seed(std::random_device{}());
        auto dice_rand = std::bind(std::uniform_int_distribution<l_int>(p0,p1),mt19937(seed));
        l_int xx = dice_rand();
        l_int p_ini = Fock_E_n[xx].second;
        //l_int p_ini = basis.get_state(Fock_E_n[xx].second);
        pair<int, l_int> aux = make_pair(xx, p_ini);
        target_ind_Fock.push_back(aux);
    }
    cout << target_ind_Fock.size() << endl;

    // print time vec to file
    Vec_fwrite_double("evo_time_vec.bin", Params.time_vec, Params.evo_time_steps);
    // for each intial state 
    for (int i_p = 0; i_p < target_ind_Fock.size(); i_p++)
    {
        // 0) preliminaries: 
        //int prt_ind = target_ind_Fock[i_p].first;
        int prt_ind = i_p;
        l_int p_ind = target_ind_Fock[i_p].second;
        l_int p_state = basis.get_state(p_ind);
        //p_ind = 11;
        double E_real = H_diag[p_ind];
        // print initial state information, p_state, energy density
        char fini[80];
        sprintf(fini, "IniStateInfo_ind_state_eng_epsi_e%0.2f_ind%d.dat", epsilon, prt_ind);
        ofstream ofini(fini);
        ofini << p_ind << endl;
        ofini << p_state << endl;	// this is a binary number
        ofini << setprecision(14) << E_real << endl;
        ofini << setprecision(14) << (E_real - E_min)/E_band << endl;
        ofini.close();

        Calc_Sparse_Hsquare(p_ind, epsilon, prt_ind);

        Dyn_DataStruc dyn_data;
        dyn_data.Initialize(Params, Dim, prt_ind, epsilon);

        // |<alpha|p>|^2
        if ('y' == Params.de_IG || 'y' == Params.prt_IG_p_aa || 'y' == Params.de_IGsq || 'y' == Params.prt_IGsq_p_aa || 'y' == Params.prt_Csq_p_alpha)
        {
            Get_Csq_p_alpha(p_ind, dyn_data.Csq_p_alpha);
            // width can be obtained by |<alpha|p>|^2
            double p_Width = Get_Width_p(E_real, dyn_data.Csq_p_alpha);
            char fname[80];
            sprintf(fname, "width_e%0.2f_ind%d.dat", epsilon, prt_ind);
            ofstream ofw(fname);
            ofw << setprecision(14) << p_Width;
            ofw.close();
        }

        // <q|IG(p)|q>
        if ('y' == Params.de_IG || 'y' == Params.evo_IG)
        {
            Get_IG_p_q(basis, p_state, dyn_data.IG_p_q);
        }

        // <alpha|IG(p)|alpha>
        if ('y' == Params.de_IG || 'y' == Params.prt_IG_p_aa) {
            Get_O_p_aa(dyn_data.IG_p_q, dyn_data.IG_p_aa);
        }

        // <q|IGsq(p)|q>
        if ('y' == Params.de_IGsq || 'y' == Params.evo_IGsq)
        {
            Get_IGsq_p_q(basis, p_state, dyn_data.IGsq_p_q);
        }

        // <alpha|IGsq(p)|alpha>
        if ('y' == Params.de_IGsq || 'y' == Params.prt_IGsq_p_aa) {
            Get_O_p_aa(dyn_data.IGsq_p_q, dyn_data.IGsq_p_aa);
        }

        // 1) compute diagonal ensemble (DE) quantities 
        if ('y' == Params.de_IG)
        {
            double IG_DE_p = cblas_ddot(Dim, dyn_data.Csq_p_alpha, 1, dyn_data.IG_p_aa, 1);
            char fname[80];
            sprintf(fname, "DE_IG_e%0.2f_ind%d.dat", epsilon, prt_ind);
            ofstream ofw(fname);
            ofw << setprecision(14) << IG_DE_p;
            ofw.close();
        }

        if ('y' == Params.de_IGsq)
        {
            double IGsq_DE_p = cblas_ddot(Dim, dyn_data.Csq_p_alpha, 1, dyn_data.IGsq_p_aa, 1);
            char fname[80];
            sprintf(fname, "DE_IGsq_e%0.2f_ind%d.dat", epsilon, prt_ind);
            ofstream ofw(fname);
            ofw << setprecision(14) << IGsq_DE_p;
            ofw.close();
        }

        //if ('y' == Params.de_ExtremeP)
        if ('y' == Params.de_mz)
        {
            double* mz_ik = new double[LatticeSize * Dim];
            Vec_fread_double("mz_ik.bin", mz_ik, LatticeSize * Dim);
            double* mz_de = new double[LatticeSize];
            for (int site_i = 0; site_i < LatticeSize; site_i++)
            {
                mz_de[site_i] = cblas_ddot(Dim, dyn_data.Csq_p_alpha, 1, &mz_ik[site_i], LatticeSize);
            }
            double ExtremeP_DE = Get_ExtremeP_from_mzVec(mz_de);
            delete[]mz_ik;
            //
            char fname[80];
            sprintf(fname, "DE_mzi_e%0.2f_ind%d.dat", epsilon, prt_ind);
            ofstream ofw(fname);
            for (int site_i = 0; site_i < LatticeSize; site_i++)
            {
                ofw << setprecision(14) << mz_de[site_i] << endl;
            }
            ofw.close();
            delete[]mz_de;
            //
            /*
            char fname1[80];
            sprintf(fname1, "DE_ExtremeP_e%0.2f_ind%d.dat", epsilon, prt_ind);
            ofstream of1(fname1);
            of1 << setprecision(14) << ExtremeP_DE;
            of1.close();
            */
        }

        // 2) compute real time quantities
        Calc_evolution_p(basis, p_ind, dyn_data);

        // print all dynamics quantities
        dyn_data.PrintDynResults(Params);

        // 
        dyn_data.ReleaseSpace(Params);
    }
}

/*
l_int Hmatrix::Get_TargetFock(const double& Target_E, vector <pair<double, l_int> >& Fock_E_n) {
    for (l_int p = 0; p < Dim; p++) {
        if (Fock_E_n[p].first > Target_E) {
            return p;
        }
    }
    return -1;
}
*/

l_int Hmatrix::Get_TargetFock_left(const double& Target_E, vector <pair<double, l_int> >& Fock_E_n) {
    for (l_int p = 0; p < Dim; p++) {
        if (Fock_E_n[p].first > Target_E) {
            return p-1;
        }
    }
    return -1;
}

l_int Hmatrix::Get_TargetFock_right(const double& Target_E, vector <pair<double, l_int> >& Fock_E_n) {
    for (l_int p = 0; p < Dim; p++) {
        if (Fock_E_n[p].first > Target_E) {
            return p;
        }
    }
    return -1;
}

void Hmatrix::Get_IG_p_q(Basis& basis, const l_int& num_p, double* IG_p_q) {
    double LB = LatticeSize - NoParticles;
    for (int q = 0; q < Dim; q++) {
        int num_q = basis.get_state(q);
        int N1 = numOfBit1(num_p & num_q);
        int N0 = NoParticles - N1;
        IG_p_q[q] = (double)(N1) / (double)(NoParticles)-(double)(N0) / LB;
    }
}

void Hmatrix::Get_IGsq_p_q(Basis& basis, const l_int& num_p, double* IGsq_p_q) {
    double* vec_a = new double[LatticeSize];
    double* coeff_a = new double[LatticeSize];
    for (int i = 0; i < LatticeSize; i++) {
        vec_a[i] = (num_p >> i) & 1;
        coeff_a[i] = vec_a[i] ? 1.0 / NoParticles : -1.0 / (LatticeSize - NoParticles);
    }
    double* auxij = new double[LatticeSize * LatticeSize];
    for (int i = 0; i < LatticeSize; i++) {
        for (int j = 0; j < LatticeSize; j++) {
            auxij[i * LatticeSize + j] = coeff_a[i] * coeff_a[j];
        }
    }
    for (int q = 0; q < Dim; q++) {
        int num_q = basis.get_state(q);
        double aux = 0;
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = 0; j < LatticeSize; j++) {
                aux += auxij[i * LatticeSize + j] * ((num_q >> i) & 1) * ((num_q >> j) & 1);
            }
        }
        IGsq_p_q[q] = aux;
    }
    delete[]auxij;
    delete[]vec_a;
    delete[]coeff_a;
}

void Hmatrix::Get_Csq_p_alpha(const l_int& p_ini, double* Csq_p_alpha) {
    vdMul(Dim, &Hmat[p_ini * Dim], &Hmat[p_ini * Dim], Csq_p_alpha);
}

// <alpha|O(p)|alpha> = \sum_q <alpha|q> <q|O(p)|q> <q|alpha> = \sum_q |<q|alpha>|^2 O_p_q
void Hmatrix::Get_O_p_aa(double* O_p_q, double* O_p_aa) {
    for (l_int k = 0; k < Dim; k++) {
        double* wf = new double[Dim];
        //vdPackI(Dim, &Hmat[k], Dim, wf_q);
        H_fetch_eigvec(k, wf);
        vdMul(Dim, wf, wf, wf);
        O_p_aa[k] = cblas_ddot(Dim, wf, 1, O_p_q, 1);
        delete[]wf;
    }
}

double Hmatrix::Get_O_p_DE(double* IG_p_q, double* Csq_p_alpha, double* O_p_aa) {
    return cblas_ddot(Dim, Csq_p_alpha, 1, O_p_aa, 1);
}

double Hmatrix::Get_Width_p(const double& E_real, double* Csq_p_alpha) {
    double* E_shift = new double[Dim];
    for (int k = 0; k < Dim; k++) {
        E_shift[k] = E_real - spec[k];
    }
    vdMul(Dim, E_shift, E_shift, E_shift);
    double p_Width = sqrt(cblas_ddot(Dim, E_shift, 1, Csq_p_alpha, 1));
    delete[]E_shift;
    return p_Width;
}

void Hmatrix::Calc_evolution_p(Basis& basis, const int& p, Dyn_DataStruc& dyn_data) {
    int t_len = Params.evo_time_steps;
#pragma omp parallel for schedule(dynamic)
    for (int t_slice = 0; t_slice < t_len; t_slice++) {
        double* ket_real = new double[Dim];
        double* ket_imag = new double[Dim];
        for (int k = 0; k < Dim; k++) {
            ket_real[k] = cos(-Params.time_vec[t_slice] * spec[k]);
            ket_imag[k] = sin(-Params.time_vec[t_slice] * spec[k]);
        }
        vdMul(Dim, ket_real, &Hmat[p * Dim], ket_real);
        vdMul(Dim, ket_imag, &Hmat[p * Dim], ket_imag);

        double* ket_real_1 = new double[Dim];
        double* ket_imag_1 = new double[Dim];
        cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Hmat, Dim, ket_real, 1, 0, ket_real_1, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Hmat, Dim, ket_imag, 1, 0, ket_imag_1, 1);
        vdSqr(Dim, ket_real_1, ket_real_1);
        vdSqr(Dim, ket_imag_1, ket_imag_1);
        // psi_t sq
        vdAdd(Dim, ket_real_1, ket_imag_1, ket_real_1);
        //Calc_Sparse_Hsquare(p, 0, 1);

        // overlap, |<Psi(t)|Psi(0)>|
        dyn_data.wf0_wft_inner[t_slice] = sqrt(ket_real_1[p]);
        // observables 
        if ('y' == Params.evo_IG)
            dyn_data.IG_p_t[t_slice] = cblas_ddot(Dim, ket_real_1, 1, dyn_data.IG_p_q, 1);

        if ('y' == Params.evo_IGsq)
            dyn_data.IGsq_p_t[t_slice] = cblas_ddot(Dim, ket_real_1, 1, dyn_data.IGsq_p_q, 1);

        if ('y' == Params.evo_IPR)
            dyn_data.IPR_p_t[t_slice] = Compute_IPR_wfsq(ket_real_1);

        if ('y' == Params.evo_SE)
            dyn_data.SE_p_t[t_slice] = Compute_Shannon_Entropy_wfsq(ket_real_1);

        //if ('y' == Params.evo_ExtremeP)
        if ('y' == Params.evo_mzit)
        {
            double* mz_vec = new double[LatticeSize];
            Get_mzi_wfsq(basis, ket_real_1, mz_vec);
            //dyn_data.ExtremeP_p_t[t_slice] = Get_ExtremeP_from_mzVec(mz_vec);
            // debug
            /*
               l_int p_state = basis.get_state(p);
               double IG_t = 0;
               for (int site_i = 0; site_i < LatticeSize; site_i++) {
               IG_t += ((p_state >> site_i) & 1) ? ((mz_vec[site_i]+0.5) / NoParticles) : (- (mz_vec[site_i] + 0.5) / (LatticeSize - NoParticles));
               }
               if ((IG_t - dyn_data.IG_p_t[t_slice]) > 1e-8) {
               cout << "Error! " << "IG: " << dyn_data.IG_p_t[t_slice] << "  IG from mz: " << IG_t << endl;
               }
               */
            //
            for (int site_i = 0; site_i < LatticeSize; site_i++) {
                dyn_data.mzi_p_t[t_slice*LatticeSize + site_i] = mz_vec[site_i];
            }
            delete[] mz_vec;
        }

        //	SE_t[t_slice] = Compute_Shannon_Entropy_wfsq(ket_real_1);
        // compute and print nit
        //Get_ni(statevector, &ni_t[t_slice * LatticeSize], ket_real_1, Dim, LatticeSize);
        //
        delete[]ket_real;
        delete[]ket_imag;
        delete[]ket_real_1;
        delete[]ket_imag_1;
    }
    // debug print 
    // cout << "debug! IG_t[0]" << IG_t[0] << endl;
    // cout << "debug! IG_t[0]" << IGsq_t[0] << endl;


}
