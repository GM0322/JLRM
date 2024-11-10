import cupy
from cupy.cuda import runtime
from cupy.cuda import texture
import numpy as np

source_texref ="""

extern "C"{
#define PI 3.1415926

texture<float, 1, cudaReadModeElementType> texFP;
texture<float, 2, cudaReadModeElementType> texImage;

__global__ void fGetResiduals(float* d_fResidualsData,
	float* d_fFPData,
	int nBins,
	double fSod,
	double fOdd,
	double fCellSize,
	double fPixelSize,
	double fFovRadius,
	double fCosLambda,
	double fSinLambda,
	int nView,
	double fOffSet,
	double fAngleOfSlope)
{
	int nB = blockIdx.x * blockDim.x + threadIdx.x;

	if (nB < nBins)
	{
		float fCosAngle = cosf(fAngleOfSlope + PI / 2.0f);
		float fSinAngle = sinf(fAngleOfSlope + PI / 2.0f);

		// 探测器端点坐标
		float fPointx = -fOdd + (fOffSet - float(nBins) / 2.0f * fCellSize + (float(nB) + 0.5f) * fCellSize) * fCosAngle;
		float fPointy = (fOffSet - float(nBins) / 2.0f * fCellSize + (float(nB) + 0.5f) * fCellSize) * fSinAngle;

		//射线源坐标
		float fSx = fSod;
		float fSy = 0.0f;

		float fVecx = fPointx - fSx;
		float fVecy = fPointy - fSy;
		float fVecLength = sqrt(fVecx * fVecx + fVecy * fVecy);

		fVecx = fVecx / fVecLength;
		fVecy = fVecy / fVecLength;

		float fSamplePtx = fSx + (fSod - fFovRadius) * fVecx;
		float fSamplePty = fSy + (fSod - fFovRadius) * fVecy;

		float fProjectionValue = 0.0f;
		int nNumOfStep = ceil(2.0f * fFovRadius / (0.5f * fPixelSize));
		for (int i = 0; i < nNumOfStep; ++i)
		{
			fSamplePtx += fVecx * fPixelSize * 0.5f;
			fSamplePty += fVecy * fPixelSize * 0.5f;

			//采样点做旋转变换
			float fRSamplePtx = fSamplePtx * fCosLambda + fSamplePty * (-fSinLambda);
			float fRSamplePty = fSamplePtx * fSinLambda + fSamplePty * fCosLambda;

			//几何坐标转换为纹理坐标
			float fAddressx = (fRSamplePtx + fFovRadius) / fPixelSize;
			float fAddressy = (fRSamplePty + fFovRadius) / fPixelSize;

			fProjectionValue += tex2D(texImage, fAddressx, fAddressy);
		}
		fProjectionValue *= fPixelSize * 0.5f;

		float fFactor = 2.0f * fFovRadius;
		d_fResidualsData[nB] = (d_fFPData[nB + nBins * nView] - fProjectionValue) / fFactor;
	}
	__syncthreads();
}

__global__ void AssignResidualError_kernel(float* d_fImageData,
	int nWidth,
	int nHeight,
	int nBins,
	double fSod,
	double fOdd,
	double fCellSize,
	double fPixelSize,
	double fCosLambda,
	double fSinLambda,
	double fOffSet,
	double fAngleOfSlope,
	double relax_factor)
{
	unsigned int  w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  h = blockIdx.y * blockDim.y + threadIdx.y;
	if (h < nHeight && w < nWidth)
	{
		//由grid中的索引转换到系统坐标系下的坐标
		float fPointx = -float(nWidth) / 2.0f * fPixelSize + 0.5f * fPixelSize + float(w) * fPixelSize;
		float fPointy = -float(nHeight) / 2.0f * fPixelSize + 0.5f * fPixelSize + float(h) * fPixelSize;

		float fRX1 = fPointx * fCosLambda + fPointy * fSinLambda;
		float fRX2 = -fPointx * fSinLambda + fPointy * fCosLambda;

		float fCosAngle = cosf(fAngleOfSlope + PI / 2.0f);
		float fSinAngle = sinf(fAngleOfSlope + PI / 2.0f);

		float fIndexx = -fOffSet + (fSod + fOdd) * fRX2 / (fSinAngle * (fSod - fRX1) + fCosAngle * fRX2);

		fIndexx = fIndexx / fCellSize + float(nBins) / 2.0f - 0.5f;
		d_fImageData[h * nWidth + w] += relax_factor*tex1D(texFP, fIndexx + 0.5f);

		float fTempR = (w - float(nWidth / 2.0f) - 0.5f) * (w - float(nWidth) / 2.0f - 0.5f) + (h - float(nHeight) / 2.0f - 0.5f) * (h - float(nHeight) / 2.0f - 0.5f);
		fTempR = sqrt(fTempR);
		//将负值和视野半径之外的值设为零(d_fImageData[h * nWidth + w] < 0.0f) || 
		if (fTempR > float(1.414*nWidth / 2.0f))
			d_fImageData[h * nWidth + w] = 0.0f;
	}
	__syncthreads();
}
}
"""

def ART2D(p,sp,rp, x0_):
  x0 = x0_.copy()
  block1D = (8, 1)
  grid1D = ((sp['bins'] + block1D[0] - 1) // block1D[0], 1)
  block2D = (8, 8)
  grid2D = ((rp['nSize'] + block2D[0] - 1) // block2D[0],
            (rp['nSize'] + block2D[1] - 1) // block2D[1])
  mod = cupy.RawModule(code=source_texref)
  fGetResiduals = mod.get_function('fGetResiduals')
  AssignResidualError = mod.get_function('AssignResidualError_kernel')
  channelDescImg = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
  cuArrayImg = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
  resourceDescImg = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayImg)
  address_modeImg = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
  texDescImg = texture.TextureDescriptor(address_modeImg, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
  texture.TextureReference(mod.get_texref('texImage'), resourceDescImg, texDescImg)

  # 1D texture
  channelDesc1D = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
  cuArray1D = texture.CUDAarray(channelDesc1D, sp['bins'])
  resourceDesc1D = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArray1D)
  address_mode1D = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
  texDesc1D = texture.TextureDescriptor(address_mode1D, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
  texture.TextureReference(mod.get_texref('texFP'), resourceDesc1D, texDesc1D)

  d_fResidualsData = cupy.zeros(sp['bins'], cupy.float32)
  order = np.random.permutation(sp['views'])
  for iter in range(1):
    for v in range(sp['views']):
      nView = order[v]
      fLambda = 2.0 * np.pi / float(sp['numAngle']) * float(nView + sp['start'])
      fCosLambda = np.cos(fLambda)
      fSinLambda = np.sin(fLambda)
      cuArrayImg.copy_from(x0)
      getErrArgs = (d_fResidualsData, p, sp['bins'], sp['sod'], sp['odd'], sp['cellsize'], rp['pixelSize'],
                    rp['pixelSize']*rp['nSize']/2.0, fCosLambda, fSinLambda, nView, 0.0, 0.0)
      fGetResiduals(grid1D, block1D, getErrArgs)
      cuArray1D.copy_from(d_fResidualsData)
      AssignResidualErrorArgs = (x0, rp['nSize'], rp['nSize'], sp['bins'], sp['sod'], sp['odd'], sp['cellsize'],
                                 rp['pixelSize'], fCosLambda, fSinLambda, 0.0, 0.0, 0.5)
      AssignResidualError(grid2D, block2D, AssignResidualErrorArgs)
  return x0
