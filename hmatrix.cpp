#include "hmatrix.h"
using namespace std;
#include <omp.h>

Hmatrix::Hmatrix()
{
    Params.SetDefault();
    Params.Read_input();
    LatticeSize = Params.LatticeSize;
    NoParticles = Params.NoParticles;

    Basis basis(Params.LatticeSize, Params.NoParticles);
    Dim = basis.get_Dim();

    l_int matlen = Dim * Dim;
    // 
    cout << "Dim = " << Dim <<endl;
    cout << "Dim*Dim = " << matlen <<endl;
    cout << "Estimated memory cost of Hmat: " << Dim*Dim*8/1e9 << " GB" << endl;

    Hmat = new double[matlen];
    H_diag = new double[Dim];
    spec = new double[Dim];

    // Build Hamiltonian matrix -------------------------
    cout << setw(24) << "Step" << setw(24) << "time(s) " << endl;
    int time0 = time(0);

    int lattice_type = 0;
    // SC qubits
    if (0 == lattice_type)
    {
        cout << "Lattice: SC qubits" << endl << endl;
        Calc_CoeffMat_SCQubits();
        Build_HamMat_SCQubits(basis);
    }

    // XXZ model
    if (1 == lattice_type)
    {
        cout << "Lattice: XXZ chain with PBCs" << endl << endl;
        Build_HamMat_XXZ(basis);
    }

    // general XXZ model, with site-dependent coupings 
    if (2 == lattice_type)
    {
        cout << "Lattice: XXZ models with input Jij" << endl << endl;
        Build_HamMat_XXZ_general(basis);
    }

    int timess = time(0);
    cout << setw(24) << "Get_Hmat" << setw(24) << timess - time0 << endl;

    // ------------------------------------------- 
    Check_Mat_Conj(Hmat, Dim);
    // store the sparse form of H for later use
    Matrix_dense2csr_upper();
    cout << "No. of nonzeros: nnz = " << nnz << endl;

    //Vec_fwrite_double("Hmat.bin", Hmat, Dim * Dim);
    /*
    // <H^2> test with dense matrix
    for (int i = 0; i < Dim; i++)
    {
        for (int j = 0; j < Dim; j++)
        {
            cout << setw(12) << setprecision(4) << Hmat[i * LatticeSize + j];
        }
        cout << endl;
    }
    

    int p = 11;
    double *psi0 = new double[Dim];
    for (int i = 0; i < Dim; i++) psi0[i] = 0;
    psi0[p] = 1;
    double* psi1 = new double[Dim];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Hmat, Dim, psi0, 1, 0, psi1, 1);
    cout << "dense matrix, <p|H|p> " << cblas_ddot(Dim, psi0, 1, psi1, 1) << endl;
    cout << "dense matrix, (<p|H*)H|p> " << cblas_ddot(Dim, psi1, 1, psi1, 1) << endl;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Hmat, Dim, psi1, 1, 0, psi1, 1);
    cout << "dense matrix, (<p|HH|p> " << cblas_ddot(Dim, psi0, 1, psi1, 1) << endl;
    */

    //cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Hmat, Dim, psi1, 1, 0, psi1, 1);

    // Full matrix diagonalization 
    char evd_jobz;
    if ('y' == Params.calc_eigvec)
        evd_jobz = 'V';
    else
        evd_jobz = 'N';
    // 
    cout << "diagonaling..." << endl;
    MatrixEvd(LAPACK_ROW_MAJOR, evd_jobz, 'U', Dim, Hmat, Dim, spec);
    // caution! Fortran interface gives different (row & col prolblem results! to be solved later
    //MatrixEvd_Fortran(LAPACK_ROW_MAJOR, evd_jobz, 'U', Dim, Hmat, Dim, spec);
    FILE* out_spec;
    out_spec = fopen("spec.bin", "wb");
    fwrite(spec, sizeof(double), Dim, out_spec);
    fclose(out_spec);
    int time1 = time(0);
    cout << setw(24) << "Eigs" << setw(24) << time1 - timess << endl;

    // Static quantities 
    Calc_Static_Quantities(basis);
    int time2 = time(0);
    cout << setw(24) << "Static" << setw(24) << time2 - time1 << endl;

    // Dynamic quantities
    Calc_Dynamic_Quantities(basis);
    int time3 = time(0);
    cout << setw(24) << "Dynamic" << setw(24) << time3 - time2 << endl;
}
Hmatrix::~Hmatrix()
{
    delete[]Hmat;
    delete[]H_diag;
    delete[]spec;
}

void Hmatrix::Check_Mat_Conj(double *mat, l_int dim)
{
    double aux = 0;
    for (int i = 0; i < dim; i++)
    {
        for (int j = i+1; j < dim; j++)
        {
            aux += abs(mat[i*dim+j] - mat[j*dim+i]);
        }
    }
    cout << "abs(H - H^T) = " << aux << endl;
}

void Hmatrix::Build_HamMat_XXZ(Basis& basis) 
{
    // Hamiltonian in PRL 123, 180601 (2019)
    // read in disorder if required
    hii = new double[LatticeSize];
    ifstream model_dis("input_site_disorder.dat", ios::in);
    if (model_dis) {
        std::cout << "Random onsite disorder read from input_site_disorder.dat" << endl;
        for (int i = 0; i < LatticeSize; i++) {
            model_dis >> hii[i];
        }
    }
    else {
        for (int i = 0; i < LatticeSize; i++) { hii[i] = Params.Rand_V * 2 * (ran_num() - 0.5); }
        ofstream of_disorder("input_site_disorder.dat");
        for (int i = 0; i < LatticeSize; i++) {
            of_disorder << setprecision(14) << hii[i] << endl;
        }
        of_disorder.close();
    }

    //
    int L = LatticeSize;
    int L1 = L - 1;
    l_int mat_length = Dim * Dim;
    SetToZero(Hmat, mat_length);
    SetToZero(H_diag, Dim);
    /*-------------------------------------------------------------------
      H = sum_{j} S_j^+ * S_{j+1}^-
      + g * sum_{j} S_j^z * S_{j+1}^z
      + sum_{j} h_j*n_j
      --------------------------------------------------------------------*/
    //	omp_set_num_threads(NUMTHREADS);
#pragma omp parallel for schedule(dynamic)
    for (l_int k = 0; k < Dim; k++) {
        l_int state_k, state_k1;
        state_k = basis.get_state(k);
        state_k1 = state_k >> 1;
        if (state_k & 1) state_k1 += (1 << L1);		// PBCs
        // count anti-ferro magnetic bonds : number of 01 neighbours
        l_int state_xor = state_k ^ state_k1;
        int n01 = numOfBit1(state_k ^ state_k1);

        // diagonal term ---------------------------------------------------	
        l_int arrayindex = k + k * Dim;
        // +g*sum_{j}S^z_{j} * S^z_{j+1}
        Hmat[arrayindex] += Params.XXZ_g * 0.25 * (L - n01 - n01);
        // sum_{j}hii[j]*n_j;
        for (int i = 0; i < LatticeSize; i++) {
            Hmat[arrayindex] += hii[i] * ((state_k >> i) & 1);
        }
        H_diag[k] = Hmat[arrayindex];
        // off-diagonal term -----------------------------------------------
        // \sum_j t_j*(c_i^\dagger c_j + h.c.)
        if (n01 > 0) {
            l_int state_l;
            int* b = new int[n01];
            findBit1(state_xor, n01, b);
            for (int i = 0; i < n01; i++) {
                // consider only upper H, l > k
                if (b[i] == L1) {
                    state_l = ((state_k ^ (1 << L1)) ^ 1);
                    l_int l = basis.get_index(state_l);
                    Hmat[k + l * Dim] += 0.5;
                }
                else {
                    state_l = (state_k ^ (3 << b[i]));
                    l_int l = basis.get_index(state_l);
                    Hmat[k + l * Dim] += 0.5;
                }
            }
            delete[]b;
        }
        // debug
        // PrintHMatrix(0);
    }
}

void Parameters::Read_input_SCQubits()
{
    // read in couplings 
    Qubits_Jij = new double[LatticeSize * LatticeSize];
    ifstream model_Jij("Jij.in", ios::in);
    if (model_Jij) {
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = 0; j < LatticeSize; j++) {
                model_Jij >> Qubits_Jij[i * LatticeSize + j];
            }
        }
        model_Jij.close();
    }
    else {
        cout << "Error: no Jij.in found! Exit!" << endl;
        exit(-1);
    }
    // cut couplings to nth neighbors, and consider a "ring" structure
    GetParaFromInput_char("input.in", "Qubits_cut_longrange", Qubits_cut_longrange);
    GetParaFromInput_int("input.in", "Qubits_cut_to_n", Qubits_cut_to_n);
    if ('y' == Qubits_cut_longrange) {
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = 0; j < LatticeSize; j++) {
                int aux = abs(i - j);
                int dist = aux < LatticeSize - aux ? aux : LatticeSize - aux;
                if (dist > Qubits_cut_to_n) {
                    Qubits_Jij[i * LatticeSize + j] = 0;
                }
            }
        }
        // print cut couplings 
        char fname[40];
        sprintf(fname, "Jij_cut%d.dat", Qubits_cut_to_n);
        ofstream of_Jcut(fname);
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = 0; j < LatticeSize; j++) {
                of_Jcut << setw(10) << setprecision(4) << Qubits_Jij[i * LatticeSize + j];
            }
            of_Jcut << endl;
        }
        of_Jcut.close();
    }

    // read in disorder if required
    Rand_Vec = new double[LatticeSize];
    ifstream model_dis("input_site_disorder.dat", ios::in);
    if (model_dis) {
        std::cout << "Random onsite disorder read from input_site_disorder.dat" << endl;
        for (int i = 0; i < LatticeSize; i++) {
            model_dis >> Rand_Vec[i];
        }
    }
    else {
        for (int i = 0; i < LatticeSize; i++) { Rand_Vec[i] = Rand_V * 2 * (ran_num() - 0.5); }
    }
}

void Hmatrix::Calc_CoeffMat_SCQubits()
{
    cout << "Hamiltonian: Superconducting qubits (with disorder and/or stark potental):" << endl;
    cout << "H = sum_{i < j}J_{ij}[b_i^+ b_j^- + b_i^- b_j^+] ";
    cout << "+ sum_{i} h_i b_i^+b_i^-" << endl;
    cout << "h_i is the potential of the ith spin (consider both disorder and stark potential)" << endl;
    cout << "stark potential: " << "-gamma*(i-x0)+alpha*(i-x0)^2" << endl;
    cout << "x0 = (L-1)/2" << endl;

    Params.Read_input_SCQubits();
    //
    double rescale_coeff = 2 * PI / 1000;			// initial unit: MHz/2/pi, after: GHz, corresponding time scale: ns
    double x0 = 0.5 * (LatticeSize - 1);		// center site 
    //
    hii = new double[LatticeSize];
    //
    for (int i = 0; i < LatticeSize; i++) {
        hii[i] = Params.Qubits_Jij[i * LatticeSize + i] * rescale_coeff;			// g[i]^2 /Delta
        hii[i] += Params.Rand_Vec[i] * rescale_coeff;			// disorder term: hii in [-V,V]
        hii[i] += -(i - x0) * Params.Stark_gamma * rescale_coeff;	// Stark term: -gamma*i
        hii[i] += (i - x0) * (i - x0) * Params.Stark_alpha * rescale_coeff;	// Stark term: alpha * i^2
    }
    //
    Jij = new double[LatticeSize * LatticeSize];
    for (int i = 0; i < LatticeSize; i++) {
        for (int j = 0; j < LatticeSize; j++) {
            Jij[i * LatticeSize + j] = Params.Qubits_Jij[i * LatticeSize + j] * rescale_coeff;
        }
    }
    //
    std::cout << "input Random onsite potential (MHz/2/pi):" << endl;
    ofstream of_disorder("input_site_disorder.dat");
    for (int i = 0; i < LatticeSize; i++) {
        std::cout << setprecision(8) << Params.Rand_Vec[i] << endl;
        of_disorder << setprecision(8) << Params.Rand_Vec[i] << endl;
    }
    of_disorder.close();
    std::cout << "onsite potential in code (GHz):" << endl;
    for (int i = 0; i < LatticeSize; i++) {
        std::cout << setprecision(8) << hii[i] << endl;
    }
}

void Hmatrix::Build_HamMat_SCQubits(Basis& basis)
{
    l_int mat_length = Dim * Dim;
    SetToZero(Hmat, mat_length);
    SetToZero(H_diag, Dim);
    //	omp_set_num_threads(NUMTHREADS);
    //	#pragma omp parallel for schedule(dynamic)
    /*-------------------------------------------------------------------
      H = sum_{i < j}J_{ij}[b_i^+ b_j^- + b_i^- b_j^+]
      + sum_{i} (h_i + dh_i) (b_i^+b_i^- - 1/2)
      --------------------------------------------------------------------*/
    for (unsigned long long k = 0; k < Dim; k++) {
        double H_aux;
        l_int Num_k, Num_k1;
        Num_k = basis.get_state(k);
        // diagonal term ---------------------------------------------------	
        l_int arrayindex = k + k * Dim;
        // sum_{j}hii[j]*n_j;
        for (int i = 0; i < LatticeSize; i++) Hmat[arrayindex] += hii[i] * (((Num_k >> i) & 1));

        // debug, diagonal term 
        H_diag[k] = Hmat[arrayindex];
        // off-diagonal term -----------------------------------------------
        // - t \sum_{i \neq j} t_{ij} [c_i^\dagger c_j + H.c.]
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = i + 1; j < LatticeSize; j++) {
                if (0 != Jij[i * LatticeSize + j]) {
                    // if state on i and j are different 
                    if ((((Num_k >> i) & 1) ^ ((Num_k >> j) & 1))) {
                        // then flip the states on i and j 
                        l_int Num_l = Num_k ^ (1 << i);
                        Num_l = Num_l ^ (1 << j);
                        l_int state_l = basis.get_index(Num_l);
                        Hmat[k * Dim + state_l] += Jij[i * LatticeSize + j];
                    }
                }
            }
        }
    }
    // debug
    // PrintHMatrix(0);
}

void Hmatrix::Build_HamMat_XXZ_general(Basis& basis)
{

    Params.Read_input_SCQubits();
    //
    double rescale_coeff = 2 * PI / 1000;			// initial unit: MHz/2/pi, after: GHz, corresponding time scale: ns
    double x0 = 0.5 * (LatticeSize - 1);		// center site 
    //
    hii = new double[LatticeSize];
    //
    for (int i = 0; i < LatticeSize; i++) {
        hii[i] = Params.Qubits_Jij[i * LatticeSize + i];
        hii[i] += Params.Rand_Vec[i];
        hii[i] += -(i - x0) * Params.Stark_gamma;
        hii[i] += (i - x0) * (i - x0) * Params.Stark_alpha;
    }
    //
    Jij = new double[LatticeSize * LatticeSize];
    for (int i = 0; i < LatticeSize; i++) {
        for (int j = 0; j < LatticeSize; j++) {
            Jij[i * LatticeSize + j] = Params.Qubits_Jij[i * LatticeSize + j];
        }
    }
    //
    std::cout << "input Random onsite potential (MHz/2/pi):" << endl;
    ofstream of_disorder("input_site_disorder.dat");
    for (int i = 0; i < LatticeSize; i++) {
        std::cout << setprecision(8) << Params.Rand_Vec[i] << endl;
        of_disorder << setprecision(8) << Params.Rand_Vec[i] << endl;
    }
    of_disorder.close();
    std::cout << "onsite potential in code (GHz):" << endl;
    for (int i = 0; i < LatticeSize; i++) {
        std::cout << setprecision(8) << hii[i] << endl;
    }

    l_int mat_length = Dim * Dim;
    SetToZero(Hmat, mat_length);
    SetToZero(H_diag, Dim);
    //	omp_set_num_threads(NUMTHREADS);
    //	#pragma omp parallel for schedule(dynamic)
    /*-------------------------------------------------------------------
      H = sum_{i < j}J_{ij}[b_i^+ b_j^- + b_i^- b_j^+]
      + sum_{i} (h_i + dh_i) (b_i^+b_i^- - 1/2)
      --------------------------------------------------------------------*/
    for (unsigned long long k = 0; k < Dim; k++) {
        double H_aux;
        l_int Num_k, Num_k1;
        Num_k = basis.get_state(k);
        // diagonal term ---------------------------------------------------	
        l_int arrayindex = k + k * Dim;
        // sum_{j}hii[j]*S^z_j;
        // sum_{ij}Jij[i,j]*S^z_i*S^z_j;
        for (int i = 0; i < LatticeSize; i++)
        {
            // hii[i]*S^z_i;
            double k_i = ((Num_k >> i) & 1);
            Hmat[arrayindex] += hii[i] * (k_i - 0.5);
            for (int j = i+1; j < LatticeSize; j++)
            {
                double k_j = ((Num_k >> j) & 1);
                Hmat[arrayindex] += 2 * Jij[i*LatticeSize + j] * (k_i - 0.5) * (k_j - 0.5);
            }
        } 
        // debug, diagonal term 
        H_diag[k] = Hmat[arrayindex];
        // off-diagonal term -----------------------------------------------
        // - t \sum_{i \neq j} t_{ij} [c_i^\dagger c_j + H.c.]
        for (int i = 0; i < LatticeSize; i++) {
            for (int j = i + 1; j < LatticeSize; j++) {
                if (0 != Jij[i * LatticeSize + j]) {
                    // if state on i and j are different 
                    if (1 == (((Num_k >> i) & 1) ^ ((Num_k >> j) & 1))) {
                        // then flip the states on i and j 
                        unsigned long long Num_l = Num_k ^ (1 << i);
                        Num_l = Num_l ^ (1 << j);
                        int state_l = basis.get_index(Num_l);
                        Hmat[k * Dim + state_l] += Jij[i * LatticeSize + j];
                    }
                }
            }
        }
    }
    // debug
    // PrintHMatrix(0);
}

inline void Hmatrix::PrintHMatrix(int aindex) 
{
    if (0 == aindex) {
        ofstream ofh("Hmatrix.dat");
        for (l_int i = 0; i < Dim; i++) {
            for (l_int j = 0; j < Dim; j++) {
                ofh << setw(12) << Hmat[i * Dim + j];
            }
            ofh << endl;
        }
        ofh.close();
    }
    else 
    {
        ofstream ofh("Hmatrix_eigenvec.dat");
        for (l_int i = 0; i < Dim; i++) {
            for (l_int j = 0; j < Dim; j++) {
                ofh << setw(12) << Hmat[i * Dim + j];
            }
            ofh << endl;
        }
    }
}

double Hmatrix::Compute_Shannon_Entropy_wfsq(const double* wfsq) {
    double auxse = 0;
    for (int j = 0; j < Dim; j++) {
        double xx = wfsq[j];
        if (xx > 1.0e-32)
            auxse += -xx * log(xx);
    }
    return auxse;
}
double Hmatrix::Compute_IPR_wfsq(const double* wfsq) {
    //
    double* auxvec = new double[Dim];
    double IPR = 1.0 / cblas_dnrm2(Dim, wfsq, 1);	// dnrm2 = sqrt(sum(wfsq^2))
    IPR *= IPR;
    return IPR;
}

void Hmatrix::Calc_Sparse_Hsquare(const l_int& p, const double& epsilon, const int& prt_ind)
{
    sparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, Dim, Dim, SMat_PointerBE, SMat_PointerBE + 1, SMat_cols, SMat_vals);
    struct matrix_descr descrA;
    //descrA.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_optimize(A);

    double* psi_0 = new double[Dim];
    for (int i = 0; i < Dim; i++) { psi_0[i] = 0; }
    psi_0[p] = 1;

    double* psi_1 = new double[Dim];

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, psi_0, 0, psi_1);

    // psi_1 is real
    double H2 = cblas_ddot(Dim, psi_1, 1, psi_1, 1);
    double H1 = cblas_ddot(Dim, psi_0, 1, psi_1, 1);

    delete[] psi_0;
    delete[] psi_1;

    char fname[80];
    sprintf(fname, "H2_H_e%0.2f_ind%d.dat", epsilon, prt_ind);
    ofstream ofw(fname);
    ofw << setprecision(14) << H2 << endl;
    ofw << setprecision(14) << H1 << endl;
    ofw.close();

    cout << p << ", H2[" << prt_ind << "] = " << H2 << ", H1[" << prt_ind << "] = " << H1 << endl;
}

// dense to sparse (csr) represetation for symmetric matrix 
void Hmatrix::Matrix_dense2csr_upper()
{
    // count number of nonzeros in the upper triangular 
    // consider all diagonal values as non zeros
    nnz = 0;
    for (lapack_int r = 0; r < Dim; r++)
    {
        nnz++;
        for (lapack_int c = r + 1; c < Dim; c++)
        {
            if (0 != Hmat[r * Dim + c])
            {
                nnz++;
            }
        }
    }

    // get cols, vals and pointerBE
    SMat_PointerBE = new lapack_int[Dim + 1];
    SMat_cols = new lapack_int[nnz];
    SMat_vals = new double[nnz];
    size_t counts = 0;
    for (lapack_int r = 0; r < Dim; r++)
    {
        SMat_PointerBE[r] = counts;
        SMat_cols[counts] = r;
        SMat_vals[counts] = Hmat[r * Dim + r];
        counts++;
        for (lapack_int c = r + 1; c < Dim; c++)
        {
            if (0 != Hmat[r * Dim + c])
            {
                SMat_cols[counts] = c;
                SMat_vals[counts] = Hmat[r * Dim + c];
                counts++;
            }
        }
        SMat_PointerBE[r + 1] = counts;
    }
}