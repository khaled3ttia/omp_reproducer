#include <omp.h>
#include "launch.h"

void saxpy(float *x, float *y, int n, float a){

    int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();

    if (i < n) 
        y[i] = a * x[i] + y[i];
}


int main(){

    int N = 1 << 20; 
    float *x, *y, *d_x, *d_y;

    x = new float[N];
    y = new float[N];

    int host = omp_get_initial_device();
    int device = omp_get_default_device();
    
    d_x = (float *)omp_target_alloc(N*sizeof(float), device);
    d_y = (float *)omp_target_alloc(N*sizeof(float), device);


    for (int i=0; i < N; i++){

        x[i] = 1.0f;
        y[i] = 2.0f;
    }

  
    omp_target_memcpy(d_x, x, N*sizeof(float), 0, 0, device, host);
    omp_target_memcpy(d_y, y, N*sizeof(float), 0, 0, device, host);

    launch<float, decltype(saxpy), saxpy>({(N+255)/256, 256}, d_x, d_y, N, 2.0f);

    synchronize();

    omp_target_free(d_x, device);
    omp_target_free(d_y, device);

    delete [] x;
    delete [] y;


}
