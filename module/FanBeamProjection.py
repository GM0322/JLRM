import cupy
from cupy.cuda import runtime
import numpy as np

source_ref = '''

extern "C"{
__host__ __device__ float maxf(float x, float y)
{
	return x > y ? x : y;
}

__global__ void fp_kernel(float* proj,float* image,int Nx,int Ny,double dx,double dy,
	double view,int bins,double du,double sod,double odd)
{	
	unsigned int bin = blockDim.x * blockIdx.x + threadIdx.x;

	if (bin > bins)return;
	float xsource = sod * cos(view);
	float ysource = sod * sin(view);
	float xdetCenter = -odd * cos(view);
	float ydetCenter = -odd * sin(view);
	float eux = -sin(view);
	float euy = cos(view);
	float u = (bin + 0.5) * du - du * bins / 2.0f;
	float xbin = xdetCenter + eux * u;
	float ybin = ydetCenter + euy * u;

	float x1 = -Nx * dx / 2.0f;
	float y1 = -Ny * dy / 2.0f;
	float xdiff = xbin - xsource;
	float ydiff = ybin - ysource;
	float xad = abs(xdiff) * dy;
	float yad = abs(ydiff) * dx;
	float raysum = 0.0f;
	if (xad > yad)
	{
		float slope = ydiff / xdiff;
		float travPixlen = dx * sqrt(1 + slope * slope);
		float yIntOld = ysource + slope * (x1 - xsource);
		int iyOld = int(floor((yIntOld - y1) / dy));
		for (int ix = 0; ix < Nx; ++ix)
		{
			float x = x1 + dx * (ix + 1.0);
			float yIntercept = ysource + slope * (x - xsource);
			int iy = int(floor((yIntercept - y1) / dy));
			if ((iy == iyOld)&&(iy >= 0) && (iy < Ny))
			{
				raysum += travPixlen * image[ix * Ny + iy];
			}
			else {
				float yMid = dy * maxf(iy, iyOld) + y1;
				float ydist1 = abs(yMid - yIntOld);
				float ydist2 = abs(yIntercept - yMid);
				float frac1 = ydist1 / (ydist1 + ydist2);
				if ((iyOld >= 0) && (iyOld < Ny))
					raysum += frac1 * travPixlen * image[iyOld + ix * Ny];
				if ((iy >= 0) && (iy < Ny))
					raysum += (1 - frac1) * travPixlen * image[iy + ix * Ny];
			}
			iyOld = iy;
			yIntOld = yIntercept;
		}
	}
	else {
		float slopeinv = xdiff / ydiff;
		float travPixlen = dy * sqrt(1.0 + slopeinv * slopeinv);
		float xIntOld = xsource + slopeinv * (y1 - ysource);
		int ixOld = int(floor((xIntOld - x1) / dx));
		for (int iy = 0; iy < Ny; ++iy)
		{
			float y = y1 + dy * (iy + 1.0f);
			float xIntercept = xsource + slopeinv * (y - ysource);
			int ix = int(floor((xIntercept - x1) / dx));
			if ((ix == ixOld) && (ix >= 0) && (ix < Nx))
				raysum += travPixlen * image[iy + ix * Ny];
			else {
				float xMid = dx * maxf(ix, ixOld) + x1;
				float xdist1 = abs(xMid - xIntOld);
				float xdist2 = abs(xIntercept - xMid);
				float frac = xdist1 / (xdist1 + xdist2);
				if ((ixOld >= 0) && ixOld < Nx)
					raysum += frac * travPixlen * image[iy + ixOld * Ny];
				if ((ix >= 0) && (ix < Nx))
					raysum += (1 - frac) * travPixlen * image[iy + ix * Ny];
			}
			ixOld = ix;
			xIntOld = xIntercept;
		}
	}
	proj[bin] = raysum;
}
}

'''


def projection(x, sp, rp):
    block1D = (8, 1)
    grid1D = ((sp['bins'] + block1D[0] - 1) // block1D[0], 1)
    mod = cupy.RawModule(code=source_ref)
    fp_kernel = mod.get_function('fp_kernel')
    p = cupy.zeros((sp['views'], sp['bins']), dtype=cupy.float32)
    for v in range(sp['views']):
        sangle = (sp['start'] + v) * 2 * np.pi / (sp['numAngle'])
        fpArgs = (p[v, ...], x, rp['nSize'], rp['nSize'], rp['pixelSize'], rp['pixelSize'],
                  sangle, sp['bins'], sp['cellsize'], sp['sod'], sp['odd'])
        fp_kernel(grid1D, block1D, fpArgs)
    return p