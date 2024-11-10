import cupy
from cupy.cuda import runtime
from cupy.cuda import texture
# import ChannelFormatDescriptor, CUDAarray, ResourceDescriptor, TextureDescriptor,TextureReference
# from utils import config
import numpy as np

source_texref = r'''

extern "C"{
#define PI 3.1415926

texture<float, 2, cudaReadModeElementType> tex_H2O;
texture<float, 2, cudaReadModeElementType> tex_Al;
texture<float, 2, cudaReadModeElementType> tex_Bone;
texture<float, 2, cudaReadModeElementType> tex_residual;

__global__ void fp_and_get_residual_eart(float* image_H2O, float* image_Al, int image_resolution, double pixel_size,
	float* spectrum, int spectrum_size,  float* muH2O, float* muAl, int mu_size,
	float* proj, int bins, int view, double start_angle, int num_angle, double cell_size,
	double sod, double odd, double dir_rotation, double offset, double object_offset_x,double object_offset_y,double angle_of_slope,
	float* d_fp_data, float* d_qphi, float* d_qtheta, float* d_qone)
{
	int bin = blockIdx.x * blockDim.x + threadIdx.x; 
	float fov_radius = pixel_size * image_resolution / 2.0f;
	float dphi = 2.0f * PI / num_angle;
	float phi = dphi * (view+start_angle) * dir_rotation;
	float cosphi = cos(phi);
	float sinphi = sin(phi);

	// 探测器方向
	float detector_dirx = cosf(angle_of_slope + phi + PI / 2.0f);
	float detector_diry = sinf(angle_of_slope + phi + PI / 2.0f);

	//射线源坐标
	float Sx = sod * cosphi;
	float Sy = sod * sinphi;

	// 当前探元坐标
	float Dx = -odd * cosphi + (offset + (bin + 0.5f - bins / 2.0f) * cell_size) * detector_dirx;
	float Dy = -odd * sinphi + (offset + (bin + 0.5f - bins / 2.0f) * cell_size) * detector_diry;

	float vecx = Dx - Sx;
	float vecy = Dy - Sy;
	float veclength = sqrt(vecx * vecx + vecy * vecy);

	vecx = vecx / veclength;
	vecy = vecy / veclength;

	float tmpptx = Sx + (sod - fov_radius) * vecx;
	float tmppty = Sy + (sod - fov_radius) * vecy;

	float projection_H2O = 0.0f;
	float projection_Al = 0.0f;
	for (int i = 0; i < image_resolution * 2; i++)
	{
		tmpptx += vecx * pixel_size * 0.5f;
		tmppty += vecy * pixel_size * 0.5f;

		float address_x = (tmpptx + fov_radius) / pixel_size;	//变为纹理坐标
		float address_y = (fov_radius + tmppty) / pixel_size;
		projection_H2O += tex2D(tex_H2O, address_x+ object_offset_x/pixel_size, address_y+ object_offset_y/pixel_size);
		projection_Al += tex2D(tex_Al, address_x + object_offset_x/pixel_size, address_y + object_offset_y/pixel_size);
	}
	projection_H2O *= pixel_size * 0.5f;
	projection_Al *= pixel_size * 0.5f;


	if (projection_H2O < 0.0f)
	{
		projection_H2O = 0.0f;
	}
	if (projection_Al < 0.0f)
	{
		projection_Al = 0.0f;
	}
	float qone = 0.0f;
	float qphi = 0.0f;
	float qtheta = 0.0f;
	for (int i = 0; i < spectrum_size; i++)
	{
		qone += 0.5*spectrum[i] * exp(-projection_H2O * muH2O[i] - projection_Al * muAl[i]);
		qphi += 0.5*spectrum[i] * muH2O[i] * exp(-projection_H2O * muH2O[i] - projection_Al * muAl[i]);
		qtheta += 0.5*spectrum[i] * muAl[i] * exp(-projection_H2O * muH2O[i] - projection_Al * muAl[i]);
	}
	if (bin < bins)
	{
		d_qone[bin] = qone;
		d_fp_data[bin] = proj[view * bins + bin] - (-log(qone));
		d_qphi[bin] = qphi / qone;
		d_qtheta[bin] = qtheta / qone;
	}
}

__global__ void fp_and_get_residual_eart3mater(float* image_H2O, float* image_Al, float* image_Bone, int image_resolution, double pixel_size,
	float* spectrum, int spectrum_size, float* muH2O, float* muAl, float* muBone, int mu_size,
	float* proj, int bins, int view, double start_angle, int num_angle, double cell_size,
	double sod, double odd, double dir_rotation, double offset, double object_offset_x, double object_offset_y, double angle_of_slope,
	float* d_fp_data, float* d_qphi, float* d_qtheta, float* d_qzeta, float* d_qone)
{
	int bin = blockIdx.x * blockDim.x + threadIdx.x;
	float fov_radius = pixel_size * image_resolution / 2.0f;

	float dphi = 2.0f * PI / num_angle;
	float phi = dphi * (view + start_angle) * dir_rotation;

	float cosphi = cos(phi);
	float sinphi = sin(phi);

	// 探测器方向
	float detector_dirx = cosf(angle_of_slope + phi + PI / 2.0f);
	float detector_diry = sinf(angle_of_slope + phi + PI / 2.0f);

	//射线源坐标
	float Sx = sod * cosphi;
	float Sy = sod * sinphi;

	// 当前探元坐标
	float Dx = -odd * cosphi + (offset + (bin + 0.5f - bins / 2.0f) * cell_size) * detector_dirx;
	float Dy = -odd * sinphi + (offset + (bin + 0.5f - bins / 2.0f) * cell_size) * detector_diry;

	float vecx = Dx - Sx;
	float vecy = Dy - Sy;
	float veclength = sqrt(vecx * vecx + vecy * vecy);

	vecx = vecx / veclength;
	vecy = vecy / veclength;

	float tmpptx = Sx + (sod - fov_radius) * vecx;
	float tmppty = Sy + (sod - fov_radius) * vecy;

	float projection_H2O = 0.0f;
	float projection_Al = 0.0f;
	float projection_Bone = 0.0f;
	for (int i = 0; i < image_resolution * 2; i++)
	{
		tmpptx += vecx * pixel_size * 0.5f;
		tmppty += vecy * pixel_size * 0.5f;

		float address_x = (tmpptx + fov_radius) / pixel_size;	//变为纹理坐标
		float address_y = (fov_radius + tmppty) / pixel_size;
		projection_H2O += tex2D(tex_H2O, address_x + object_offset_x / pixel_size, address_y + object_offset_y / pixel_size);
		projection_Al += tex2D(tex_Al, address_x + object_offset_x / pixel_size, address_y + object_offset_y / pixel_size);
		projection_Bone += tex2D(tex_Bone, address_x + object_offset_x / pixel_size, address_y + object_offset_y / pixel_size);
	}
	projection_H2O *= pixel_size * 0.5f;
	projection_Al *= pixel_size * 0.5f;
	projection_Bone *= pixel_size * 0.5;

	if (projection_H2O < 0.0f)
	{
		projection_H2O = 0.0f;
	}
	if (projection_Al < 0.0f)
	{
		projection_Al = 0.0f;
	}
	if (projection_Bone < 0.0f)
		projection_Bone = 0.0f;
	float qone = 0.0f;
	float qphi = 0.0f;
	float qtheta = 0.0f;
	float qzeta = 0.0f;
	for (int i = 0; i < spectrum_size; i++)
	{
		/////////////////////////////////////////////////////////////////////////////
		float temp = 0.5*spectrum[i] * exp(-projection_H2O * muH2O[i] - projection_Al * muAl[i] - projection_Bone * muBone[i]);
		qone += temp;
		qphi += muH2O[i] * temp;
		qtheta += muAl[i] * temp;
		qzeta += muBone[i] * temp;
	}
	if (bin < bins)
	{
		d_qone[bin] = qone;
		d_fp_data[bin] = proj[view * bins + bin] - (-log(qone));
		d_qphi[bin] = qphi / qone;
		d_qtheta[bin] = qtheta / qone;
		d_qzeta[bin] = qzeta / qone;
	}
}

__global__ void back_projection_eart(float* d_image_H2O, float* d_image_Al, float* d_qphi, float* d_qtheta, 
	int image_resolution, double sod, double odd, double cell_size, double pixel_size,
	int bins, int view, double start_angle, int num_angle,
	double relax_H2O, double relax_Al, 
	float* d_qone, double fDirRotation, double offset, double object_offset_x, double object_offset_y, double angle_of_slope)
{

	int w = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	if (w >= image_resolution || h >= image_resolution)return;

	float fov_radius = pixel_size * image_resolution / 2.0f;

	float dphi = 2.0f * PI / num_angle;
	float phi = dphi * (view + start_angle) * fDirRotation;

	float cosphi = cos(phi);
	float sinphi = sin(phi);

	//由pixel序号求pixel局部坐标系中的坐标
	float x = (w + 0.5f - image_resolution / 2.0f) * pixel_size - object_offset_x;
	float y = (h + 0.5f - image_resolution / 2.0f) * pixel_size - object_offset_y;

	//求pixel顺时针旋转phi角度后在全局坐标系中的坐标
	float rot_x = x * cosphi + y * sinphi;
	float rot_y = -x * sinphi + y * cosphi;

	float fCosAngle = cosf(angle_of_slope + PI / 2.0f);
	float fSinAngle = sinf(angle_of_slope + PI / 2.0f);

	float u = -offset + (sod + odd) * rot_y / (fSinAngle * (sod - rot_x) + fCosAngle * rot_y);

	//求探测器纹理坐标
	float xindex = u / cell_size + float(bins) / 2.0f;

	if (xindex > 0.0f && xindex < bins)
	{
		int ixindex = int(floor(xindex));

		float qphi = d_qphi[ixindex];
		float qtheta = d_qtheta[ixindex];
		float qone = d_qone[ixindex];

		float weight_H2O = (qphi) / (qtheta * qtheta + qphi * qphi);
		float weight_Al = (qtheta) / (qtheta * qtheta + qphi * qphi);


		d_image_H2O[h * image_resolution + w] += relax_H2O*weight_H2O * tex2D(tex_residual, xindex, 0.5f) / (pixel_size * image_resolution);
		d_image_Al[h * image_resolution + w] += relax_Al*weight_Al * tex2D(tex_residual, xindex, 0.5f) / (pixel_size * image_resolution);

		if (d_image_H2O[h * image_resolution + w] < 0.0f)
			d_image_H2O[h * image_resolution + w] = 0.0f;
		if (d_image_Al[h * image_resolution + w] < 0.0f)
			d_image_Al[h * image_resolution + w] = 0.0f;
	}
}

__global__ void back_projection_eart3mater(float* d_image_H2O, float* d_image_Al, float* d_image_Bone,float* d_qphi, float* d_qtheta,float*d_qzeta,
	int image_resolution, double sod, double odd, double cell_size, double pixel_size,
	int bins, int view, double start_angle, int num_angle,
	double relax_H2O, double relax_Al,double relax_Bone,
	float* d_qone, double fDirRotation, double offset, double object_offset_x, double object_offset_y, double angle_of_slope)
{

	int w = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	if (w >= image_resolution || h >= image_resolution)return;


	float fov_radius = pixel_size * image_resolution / 2.0f;

	float dphi = 2.0f * PI / num_angle;
	float phi = dphi * (view + start_angle) * fDirRotation;

	float cosphi = cos(phi);
	float sinphi = sin(phi);

	//由pixel序号求pixel局部坐标系中的坐标
	float x = (w + 0.5f - image_resolution / 2.0f) * pixel_size - object_offset_x;
	float y = (h + 0.5f - image_resolution / 2.0f) * pixel_size - object_offset_y;

	//求pixel顺时针旋转phi角度后在全局坐标系中的坐标
	float rot_x = x * cosphi + y * sinphi;
	float rot_y = -x * sinphi + y * cosphi;

	float fCosAngle = cosf(angle_of_slope + PI / 2.0f);
	float fSinAngle = sinf(angle_of_slope + PI / 2.0f);

	float u = -offset + (sod + odd) * rot_y / (fSinAngle * (sod - rot_x) + fCosAngle * rot_y);

	//求探测器纹理坐标
	float xindex = u / cell_size + float(bins) / 2.0f;


	if (xindex > 0.0f && xindex < bins)
	{
		int ixindex = int(floor(xindex));

		// 这里用邻近点插值和下面的线性插值不匹配，需要调节更好。
		float qphi = d_qphi[ixindex];
		float qtheta = d_qtheta[ixindex];
		float qzeta = d_qzeta[ixindex];
		float qone = d_qone[ixindex];

		float weight_H2O = (qphi) / (qtheta * qtheta + qphi * qphi + qzeta * qzeta);
		float weight_Al = (qtheta) / (qtheta * qtheta + qphi * qphi + qzeta * qzeta);
		float weight_Bone = qzeta / (qtheta * qtheta + qphi * qphi + qzeta * qzeta);

		d_image_H2O[h * image_resolution + w] += relax_H2O * weight_H2O * tex2D(tex_residual, xindex, 0.5f) / (pixel_size * image_resolution);
		d_image_Al[h * image_resolution + w] += relax_Al * weight_Al * tex2D(tex_residual, xindex, 0.5f) / (pixel_size * image_resolution);
		d_image_Bone[h * image_resolution + w] += relax_Bone * weight_Bone * tex2D(tex_residual, xindex, 0.5) / (pixel_size * image_resolution);

		// 图像约束项
		if (d_image_H2O[h * image_resolution + w] < 0.0f)
			d_image_H2O[h * image_resolution + w] = 0.0f;
		if (d_image_Al[h * image_resolution + w] < 0.0f)
			d_image_Al[h * image_resolution + w] = 0.0f;
		if (d_image_Bone[h * image_resolution + w] < 0.0f)
			d_image_Bone[h * image_resolution + w] = 0.0f;
		if ((w + 0.5f - image_resolution / 2.0f) * (w + 0.5f - image_resolution / 2.0f) + (h + 0.5f - image_resolution / 2.0f) * (h + 0.5f - image_resolution / 2.0f) > image_resolution * image_resolution / 4.0f)
		{
			d_image_H2O[h * image_resolution + w] = 0.0f;
			d_image_Al[h * image_resolution + w] = 0.0f;
			d_image_Bone[h * image_resolution + w] = 0.0f;
		}
	}
}

__global__ void spectrum_fp(int image_resolution, float pixel_size,
	float* spectrum_low, int spectrum_size_low, float* muH2O, float* muAl, int mu_size,
	float* proj_low, int bins_low, int view, int start_angle_low, int num_angle_low, float cell_size_low,
	float sod_low, float odd_low, float dir_rotation_low, float offset_low, 
	float object_offset_x, float object_offset_y, float angle_of_slope_low)
{
	int bin = blockIdx.x * blockDim.x + threadIdx.x;
	float fov_radius = pixel_size * image_resolution / 2.0f;

	float dphi = 2.0f * PI / num_angle_low;
	float phi = dphi * (view + start_angle_low) * dir_rotation_low;

	float cosphi = cos(phi);
	float sinphi = sin(phi);

	// 探测器方向
	float detector_dirx = cosf(angle_of_slope_low + phi + PI / 2.0f);
	float detector_diry = sinf(angle_of_slope_low + phi + PI / 2.0f);

	//射线源坐标
	float Sx = sod_low * cosphi;
	float Sy = sod_low * sinphi;

	// 当前探元坐标
	float Dx = -odd_low * cosphi + (offset_low + (bin + 0.5f - bins_low / 2.0f) * cell_size_low) * detector_dirx;
	float Dy = -odd_low * sinphi + (offset_low + (bin + 0.5f - bins_low / 2.0f) * cell_size_low) * detector_diry;

	float vecx = Dx - Sx;
	float vecy = Dy - Sy;
	float veclength = sqrt(vecx * vecx + vecy * vecy);

	vecx = vecx / veclength;
	vecy = vecy / veclength;

	float tmpptx = Sx + (sod_low - fov_radius) * vecx;
	float tmppty = Sy + (sod_low - fov_radius) * vecy;

	float projection_H2O = 0.0f;
	float projection_Al = 0.0f;
	for (int i = 0; i < image_resolution * 2; i++)
	{
		tmpptx += vecx * pixel_size * 0.5f;
		tmppty += vecy * pixel_size * 0.5f;

		float address_x = (tmpptx + fov_radius) / pixel_size;	//变为纹理坐标
		float address_y = (fov_radius + tmppty) / pixel_size;
		projection_H2O += tex2D(tex_H2O, address_x+object_offset_x/pixel_size, address_y+ object_offset_y/pixel_size);
		projection_Al += tex2D(tex_Al, address_x+ object_offset_x/pixel_size, address_y+ object_offset_y/pixel_size);
		//if (tex2D(tex_H2O, address_x, address_y) > 0.01)
		//	printf("%f\n", tex2D(tex_H2O, address_x, address_y));
	}
	projection_H2O *= pixel_size * 0.5f;
	projection_Al *= pixel_size * 0.5f;

	proj_low[view * bins_low + bin] = 0.0f;
	for (int i = 0; i < spectrum_size_low; i++)
	{
		proj_low[view*bins_low+bin] += spectrum_low[i]*exp(-projection_H2O * muH2O[i] - projection_Al * muAl[i]);
	}
	proj_low[view * bins_low + bin] = -log(proj_low[view * bins_low + bin]);

}
}
'''


def load_spectrum_and_mu(lpath, hpath):
    lowKv = np.load(lpath)
    highKv = np.load(hpath)
    sl = cupy.asarray(lowKv[1])
    # muH2O_low = lowKv[2]
    # muAl_low = lowKv[3]
    # muBone_low = lowKv[4]
    sh = cupy.asarray(highKv[1])
    muH2O = cupy.asarray(highKv[2])
    muAl = cupy.asarray(highKv[3])
    muBone = cupy.asarray(highKv[4])
    return sl, sh, muH2O, muAl, muBone


def EART2S2M(x0, proj, rp, lsp, hsp, muH2O, muAl, sl, sh, iters):
    pL = proj[0, ...]
    pH = proj[1, ...]
    H2O = x0[0, ...]#.copy()
    Al = x0[1, ...]#.copy()
    bFPDim_low = (8, 1)
    gFPDim_low = ((lsp['bins'] + bFPDim_low[0] - 1) // bFPDim_low[0], 1)
    bFPDim_high = (8, 1)
    gFPDim_high = ((hsp['bins'] + bFPDim_high[0] - 1) // bFPDim_high[0], 1)
    bBPDim = (8, 8)
    gBPDim = ((rp['nSize'] + bBPDim[0] - 1) // bBPDim[0],
              (rp['nSize'] + bBPDim[1] - 1) // bBPDim[1]);
    mod = cupy.RawModule(code=source_texref)
    back_projection_eart = mod.get_function('back_projection_eart')
    get_fp_residual_eart = mod.get_function('fp_and_get_residual_eart')
    residual_low = cupy.zeros((1, lsp['bins']), dtype=cupy.float32)
    residual_high = cupy.zeros((1, hsp['bins']), dtype=cupy.float32)
    phi_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    theta_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    phi_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)
    theta_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)
    one_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    one_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)

    channelDescImg = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayH2O = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
    cuArrayAl = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
    resourceDescH2O = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayH2O)
    resourceDescAl = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayAl)
    addressMode = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDesc = texture.TextureDescriptor(addressMode, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
    texture.TextureReference(mod.get_texref('tex_H2O'), resourceDescH2O, texDesc)
    texture.TextureReference(mod.get_texref('tex_Al'), resourceDescAl, texDesc)

    cuArrayRes = texture.CUDAarray(channelDescImg, lsp['bins'], 1)
    resourceDescRes = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayRes)
    texture.TextureReference(mod.get_texref('tex_residual'), resourceDescRes, texDesc)

    relax_H2O_low = 0.2
    relax_Al_low = 0.2
    relax_H2O_high = 0.2
    relax_Al_high = 0.2

    for iter in range(iters):
        hview = np.arange(hsp['views'])
        lview = np.arange(lsp['views'])
        np.random.shuffle(hview)
        np.random.shuffle(lview)
        for i in range(hsp['views']):
            angle_high = hview[i]
            cuArrayH2O.copy_from(H2O)
            cuArrayAl.copy_from(Al)
            fp_args = (H2O, Al, rp['nSize'], rp['pixelSize'], sh, sh.size, muH2O, muAl, sh.size,
                       pH, hsp['bins'], angle_high, hsp['start'], hsp['numAngle'],
                       hsp['cellsize'], hsp['sod'], hsp['odd'], 1.0, 0.0, 0.0, 0.0, 0.0,
                       residual_high, phi_high, theta_high, one_high)
            get_fp_residual_eart(gFPDim_high, bFPDim_high, fp_args)
            cuArrayRes.copy_from(residual_high)
            bp_args = (H2O, Al, phi_high, theta_high, rp['nSize'], hsp['sod'], hsp['odd'],
                       hsp['cellsize'], rp['pixelSize'], hsp['bins'], angle_high,
                       hsp['start'], hsp['numAngle'], relax_H2O_high, relax_Al_high,
                       one_high, 1.0, 0.0, 0.0, 0.0, 0.0)
            back_projection_eart(gBPDim, bBPDim, bp_args)
        for i in range(lsp['views']):
            angle_low = lview[i]
            cuArrayH2O.copy_from(H2O)
            cuArrayAl.copy_from(Al)
            fp_args = (H2O, Al, rp['nSize'], rp['pixelSize'], sl, sl.size, muH2O, muAl, sl.size,
                       pL, lsp['bins'], angle_low, lsp['start'], lsp['numAngle'],
                       lsp['cellsize'], lsp['sod'], lsp['odd'], 1.0, 0.0, 0.0, 0.0, 0.0,
                       residual_low, phi_low, theta_low, one_low)
            get_fp_residual_eart(gFPDim_low, bFPDim_low, fp_args)
            cuArrayRes.copy_from(residual_low)
            bp_args = (H2O, Al, phi_low, theta_low, rp['nSize'], lsp['sod'], lsp['odd'],
                       lsp['cellsize'], rp['pixelSize'], lsp['bins'], angle_low,
                       lsp['start'], lsp['numAngle'], relax_H2O_low, relax_Al_low,
                       one_low, 1.0, 0.0, 0.0, 0.0, 0.0)
            back_projection_eart(gBPDim, bBPDim, bp_args)
    recon = cupy.concatenate((cupy.resize(H2O, (1, rp['nSize'], rp['nSize'])),
                              cupy.resize(Al, (1, rp['nSize'], rp['nSize']))), axis=0)
    return recon


def EART2S3M(x0, proj, rp, lsp, hsp, muH2O, muAl, muBone, sl, sh, iters):
    pL = proj[0, ...]
    pH = proj[1, ...]
    H2O = x0[0, ...].copy()
    Al = x0[1, ...].copy()
    Bone = x0[2, ...].copy()
    bFPDim_low = (8, 1)
    gFPDim_low = ((lsp['bins'] + bFPDim_low[0] - 1) // bFPDim_low[0], 1)
    bFPDim_high = (8, 1)
    gFPDim_high = ((hsp['bins'] + bFPDim_high[0] - 1) // bFPDim_high[0], 1)
    bBPDim = (8, 8)
    gBPDim = ((rp['nSize'] + bBPDim[0] - 1) // bBPDim[0],
              (rp['nSize'] + bBPDim[1] - 1) // bBPDim[1]);

    mod = cupy.RawModule(code=source_texref)
    bp_eart3mater = mod.get_function('back_projection_eart3mater')
    get_fp_residual_eart3mater = mod.get_function('fp_and_get_residual_eart3mater')

    residual_low = cupy.zeros((1, lsp['bins']), dtype=cupy.float32)
    residual_high = cupy.zeros((1, hsp['bins']), dtype=cupy.float32)
    phi_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    theta_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    phi_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)
    theta_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)
    zeta_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    zeta_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)
    one_low = cupy.zeros(lsp['bins'], dtype=cupy.float32)
    one_high = cupy.zeros(hsp['bins'], dtype=cupy.float32)

    channelDescImg = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayH2O = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
    cuArrayAl = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
    cuArrayBone = texture.CUDAarray(channelDescImg, rp['nSize'], rp['nSize'])
    resourceDescH2O = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayH2O)
    resourceDescAl = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayAl)
    resourceDescBone = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayBone)
    addressMode = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDesc = texture.TextureDescriptor(addressMode, runtime.cudaFilterModeLinear, runtime.cudaReadModeElementType)
    texture.TextureReference(mod.get_texref('tex_H2O'), resourceDescH2O, texDesc)
    texture.TextureReference(mod.get_texref('tex_Al'), resourceDescAl, texDesc)
    texture.TextureReference(mod.get_texref('tex_Bone'), resourceDescBone, texDesc)

    cuArrayRes = texture.CUDAarray(channelDescImg, lsp['bins'], 1)
    resourceDescRes = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayRes)
    texture.TextureReference(mod.get_texref('tex_residual'), resourceDescRes, texDesc)

    relax_H2O_low = 0.5
    relax_Al_low = 0.5
    relax_Bone_low = 0.5
    relax_H2O_high = 0.5
    relax_Al_high = 0.5
    relax_Bone_high = 0.5

    for iter in range(iters):
        hview = np.arange(hsp['views'])
        lview = np.arange(lsp['views'])
        np.random.shuffle(hview)
        np.random.shuffle(lview)
        for i in range(hsp['views']):
            angle_high = hview[i]
            cuArrayH2O.copy_from(H2O)
            cuArrayAl.copy_from(Al)
            cuArrayBone.copy_from(Bone)
            fp_args = (H2O, Al, Bone, rp['nSize'], rp['pixelSize'], sh, sh.size, muH2O, muAl, muBone, sh.size,
                       pH, hsp['bins'], angle_high, hsp['start'], hsp['numAngle'],
                       hsp['cellsize'], hsp['sod'], hsp['odd'], 1.0, 0.0, 0.0, 0.0, 0.0,
                       residual_high, phi_high, theta_high, zeta_high, one_high)
            get_fp_residual_eart3mater(gFPDim_high, bFPDim_high, fp_args)
            cuArrayRes.copy_from(residual_high)
            bp_args = (H2O, Al, Bone, phi_high, theta_high, zeta_high, rp['nSize'], hsp['sod'], hsp['odd'],
                       hsp['cellsize'], rp['pixelSize'], hsp['bins'], angle_high,
                       hsp['start'], hsp['numAngle'], relax_H2O_high, relax_Al_high, relax_Bone_high,
                       one_high, 1.0, 0.0, 0.0, 0.0, 0.0)
            bp_eart3mater(gBPDim, bBPDim, bp_args)
        for i in range(lsp['views']):
            angle_low = lview[i]
            cuArrayH2O.copy_from(H2O)
            cuArrayAl.copy_from(Al)
            cuArrayBone.copy_from(Bone)
            fp_args = (H2O, Al, Bone, rp['nSize'], rp['pixelSize'], sl, sl.size, muH2O, muAl, muBone, sl.size,
                       pL, lsp['bins'], angle_low, lsp['start'], lsp['numAngle'],
                       lsp['cellsize'], lsp['sod'], lsp['odd'], 1.0, 0.0, 0.0, 0.0, 0.0,
                       residual_low, phi_low, theta_low, zeta_low, one_low)
            get_fp_residual_eart3mater(gFPDim_low, bFPDim_low, fp_args)
            cuArrayRes.copy_from(residual_low)
            bp_args = (H2O, Al, Bone, phi_low, theta_low, zeta_low, rp['nSize'], lsp['sod'], lsp['odd'],
                       lsp['cellsize'], rp['pixelSize'], lsp['bins'], angle_low,
                       lsp['start'], lsp['numAngle'], relax_H2O_low, relax_Al_low, relax_Bone_low,
                       one_low, 1.0, 0.0, 0.0, 0.0, 0.0)
            bp_eart3mater(gBPDim, bBPDim, bp_args)
    recon = cupy.concatenate((cupy.resize(H2O, (1, rp['nSize'], rp['nSize'])),
                              cupy.resize(Al, (1, rp['nSize'], rp['nSize'])),
                              cupy.resize(Bone, (1, rp['nSize'], rp['nSize']))), axis=0)
    return recon