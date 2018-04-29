#include "GenPencil.h"
#include "sparse_matrix.h"

void GetDx(unsigned id, unsigned h, unsigned w, SparseMatrix &Dx)
{
    Dx.table = NULL;
    Dx.rows = h;    Dx.cols = w;
    unsigned lower = h > w ? w : h;
    unsigned sec_lim = lower - id;
    Dx.terms = lower + sec_lim;
    Dx.table = new trituple[Dx.terms];

    unsigned zero_r = 0, zero_c = 0;
    unsigned sec_r = 0, sec_c = id;
    unsigned d_id = 0;
    while (zero_r < lower)
    {
        Dx.table[d_id].row = zero_r;
        Dx.table[d_id].col = zero_c;
        Dx.table[d_id].value = -1.0f;    
        zero_r++;   zero_c++;
        d_id++;
        if (sec_c < w)
        {
            Dx.table[d_id].row = sec_r;
            Dx.table[d_id].col = sec_c;
            Dx.table[d_id].value = 1.0f;
            sec_r++;    sec_c++;
            d_id++;
        }
    }
}

void GetDiag(double *d, unsigned length, SparseMatrix &diag)
{
    diag.table = NULL;
    diag.terms = length;
    diag.rows = length; diag.cols = length;

    for (unsigned i = 0; i < length; ++i)
    {
        diag.table[i].row = i;
        diag.table[i].col = i;
        diag.table[i].value = d[i];
    }
}

void GenPencil(const cv::Mat & input, const cv::Mat & pencil_texture, const cv::Mat & tone_map, cv::Mat & T_rst)
{
    //// OpenCV Operations ////
    // Get P
	int w = input.size().width, h = input.size().height;
	cv::Mat P;
    pencil_texture.convertTo(P, CV_64FC1);
	cv::resize(P, P, input.size());
    // Get J
    cv::Mat J;
    tone_map.convertTo(J, CV_64FC1);
    cv::resize(J, J, input.size());

    //// Matrix Operations ////
	//P.reshape(0, w * h);
    Matrix m_P;
    m_P.rows = w * h;   m_P.cols = 1;
    m_P.data = (double*)P.data;

	//cv::log(P, logP);
    Matrix m_logP;
    log(m_P, m_logP);
    
    SparseMatrix sm_logP;
    GetDiag(m_logP.data, m_logP.rows, sm_logP);

    Matrix m_J;
    m_J.rows = w * h;   m_J.cols = 1;
    m_J.data = (double*)J.data;

    Matrix m_logJ;
    log(m_J, m_logJ);

    SparseMatrix sm_Dx, sm_Dy;
    GetDx(h, w * h, w * h, sm_Dx);
    GetDx(1, w * h, w * h, sm_Dy);
    
    SparseMatrix sm_Dx_t, sm_Dy_t, Dx_mul_Dx_t, Dy_mul_Dy_t;
    SparseMatrix sm_logP_t, logP_t_mul_logP; 
    
    // get matrices transpose
    transpose(sm_Dx, sm_Dx_t);
    transpose(sm_Dy, sm_Dy_t);
    transpose(sm_logP, sm_logP_t);
    
    // theta * (Dx * Dx' + Dy * Dy') + (logP)' * logP
    mul(sm_Dx_t, sm_Dx, Dx_mul_Dx_t);
    mul(sm_Dy_t, sm_Dy, Dy_mul_Dy_t);
    mul(sm_logP_t, sm_logP, logP_t_mul_logP);

    double theta = 0.2;
    SparseMatrix Dsum;
    add(Dx_mul_Dx_t, Dy_mul_Dy_t, Dsum);
    mul(Dsum, theta, Dsum);

    SparseMatrix A;
    add(Dsum, logP_t_mul_logP, A);
    
    Matrix b;
    mul(sm_logP_t, m_logJ, b);
    
    Matrix beta;
    // TODO:pcg(A, b, 1e-6, 60);
    pcg(A, b.data, 1e-6, 60, beta.data);

    beta.rows = h;  beta.cols = w;
    m_P.rows = h;   m_P.cols = w;

    Matrix m_T;
    // TODO: power(m_P, beta, m_T);
    pow(m_P, beta, m_T);
}
