whoimport numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.linalg as LA
import math
import NanoImagingPack as nip

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
#np.set_printoptions(precision=2)

Lambda_0 = 4  # i.e. delta_x = Lambda_0/4
delta_x = Lambda_0/4

# space grid:
N = 256#np.uint(256*4//.65)+1# L = N*Lambda_0/4 (since delta_x = chosen to be 0.25 Lambda_0)
Nx = N
Nz = np.int(3/2*N)

n_x = np.linspace(-N//2,N//2,Nx)
n_z = np.linspace(-N//2,N,Nz)

X,Z = np.meshgrid(n_z,n_x, sparse=True)

''' refractive index distribution n(x,z):'''
n_0 = 1.

is_chip= True
if(is_chip):
    #%% create a wave-guide model
    # from https://refractiveindex.info/ @ lambda=600nm
    n_si = 3.969
    n_sio2 = 1.253
    n_Ta2O5 = 4.7620
    dn = np.abs(n_0-n_Ta2O5)    # maximal deviation from n_0
    
    # create model of the chip
    mask_chip = np.ones((Nx,Nz))
    
    dim_wg_height = .1625 # µm
    dim_clad_height = 5*.1625 # µm
    dim_wg_height_pix =  np.uint8(dim_wg_height//delta_x)
    dim_clad_height_pix =  np.uint8(dim_clad_height//delta_x)
    
    pos_slab_x = Nx//2
    pos_slab_z = Nz//2
    
    if(0):
        mask_chip[pos_slab_x:pos_slab_x+dim_wg_height_pix,pos_slab_z:-1]=n_Ta2O5
        mask_chip[pos_slab_x+dim_wg_height_pix:pos_slab_x+dim_wg_height_pix+dim_clad_height_pix,pos_slab_z:-1]=n_sio2
        mask_chip[pos_slab_x+dim_wg_height_pix+dim_clad_height_pix:-1,pos_slab_z:-1]=n_si
    else:
        mask_chip[pos_slab_x:pos_slab_x+1,pos_slab_z:-1]=n_Ta2O5
        mask_chip[pos_slab_x+1:pos_slab_x+6,pos_slab_z:-1]=n_sio2
        mask_chip[pos_slab_x+6:-1,pos_slab_z:-1]=n_si
    n_xz = mask_chip
    
    plt.imshow(mask_chip)
    plt.title('The Chip')
    plt.colorbar()
    plt.show()
else:
    # refractive index distribution n(x,z):
    dn = 0.3     # maximal deviation from n_0
    
    nr = 6. #4
    mask = X**2 + Z**2 <= (N/nr)**2
    n_xz = n_0 + dn*mask
    
    cp = plt.contourf(n_z,n_x,n_xz, cmap = 'Greys')
    plt.colorbar(cp)
    plt.xlabel('n_z ')
    plt.ylabel('n_x ')
    plt.axis('equal')
    plt.show()
#%%


# extended grid for absorbing layers around grid:
n_pad = 100
NNx = Nx+2*n_pad
NNz = Nz+2*n_pad
nn_x = np.arange(NNx)
nn_z = np.arange(NNz)


#%% light source
# input source: normalized Gaussian
#mySrc(myOffX,:) = exp(i*kx * (myOffY+yy([1,mysize(2:end)]))) .* exp(-abssqr(((myOffY+yy([1,mysize(2:end)]))))/(2*myWidth^2))
mywidth = 80. # in units of quarter wavelengths
x_offset = 0.
k_offset = 0.

Gx = np.exp(1j*k_offset*n_x) * np.exp(-(n_x-x_offset)**2/(2*mywidth**2))
Gx = Gx/np.sqrt(np.sum(np.sum(np.abs(Gx)**2)))
Gx *= np.exp(1j*(nip.xx(Gx.shape, freq='ftfreq')**2)*np.pi*10)
                       
plt.subplot(1, 2, 1)
plt.plot(n_x,np.real(Gx))
plt.xlabel('x')
plt.title('Re(Gx)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n_x,np.abs(Gx)**2)
plt.xlabel('x')
plt.ylabel('FWHM_x = ' + str(2*mywidth))
plt.title('I_x')
plt.grid(True)

plt.tight_layout()

print(np.sum(np.abs(Gx)**2))


plt.figure()
plt.plot(np.angle(Gx))
plt.title('Angle')

# place source into extended grid:
S = np.zeros((NNx,NNz),dtype = complex)
print(S.shape)

for j in range(Nx): 
    S[j+n_pad,n_pad] = Gx[j]
    #S[j+n_pad,NNx//2] = Gx[j]

cp = plt.contourf(np.abs(S)**2)
plt.colorbar(cp)
plt.xlabel('z ')
plt.ylabel('x ')
plt.axis('equal')
plt.title('source')
plt.colorbar
plt.show()

cp = plt.contourf(np.angle(S))
plt.colorbar(cp)
plt.xlabel('z ')
plt.ylabel('x ')
plt.axis('equal')
plt.title('source (angle)')
plt.colorbar
plt.show()


#%%

'''
important NOTE on the choice of k_0 and epsilon¶

definitions: 
k(r) = n(r)*2*pi/Lambda_0, with Lambda_0 the vacuum wavelength
k_0 in the ansatz for the field ~ exp(i k_0 z)

there are 3 conditions to be taken into account:
1) to avoid undersampling, in this approach one has to choose the spatial grid spacing delta_x <= Lambda_min/2 = (Lambda_0/n_max)/2,
choosing delta_x = Lambda_0/4 satisfies most cases (i.e. where the refractive index stays below n = 2) 

2) for convergence epsilon (regularization) has to satisfy: epsilon >= max(|k(r)^2 - k_0^2|, Eq.(11)

3) for optimal convergence one chooses k_0 to lie in the "middle" of the refractive contrast of the scattering potential V = k(r)^2 -k_0^2 - i epsilon:
k_0^2 = [min(real(k(r)^2)) + max(real(k(r)^2))]/2 (cf. p.115, discussion below Eq.(18))

'''


# specifically we choose:
n_av = (n_0 + n_0+dn)/2   # average refractive index 
k_0 = 2*np.pi/Lambda_0 *n_av
print('%.4f' %k_0)


# padded refractive index distribution
myN = np.pad(n_xz,(n_pad,n_pad),'constant', constant_values=(0, 0))   
mysize = myN.shape
print(mysize)

SlabW = n_pad  # slab width of absorber
Na = 4
alpha = n_pad * 0.055/SlabW # n_pad absorbing slices

aa = np.arange(SlabW) * alpha
PN = 0
for n in range(Na-1):
    PN = PN + aa**n /math.factorial(n)
k2 = aa**(Na-1)* np.abs(alpha)**2 *(Na-aa + 2*1j*k_0*(SlabW-aa))/(PN*math.factorial(Na))

plt.plot(np.imag(k2))

K2 = np.zeros(mysize, dtype=np.complex128)
K2[ :SlabW] += k2[::-1, np.newaxis]
K2[-SlabW:] += k2[:, np.newaxis]
K2[:, :SlabW] += k2[::-1]
K2[:, -SlabW:] += k2

plt.imshow(np.imag(K2))
plt.colorbar()


myN = n_0*np.sqrt((K2 + k_0**2)/k_0**2)  
# insert refractive index distributioin into absorber frame:
myN[n_pad:n_pad+N,n_pad:n_pad+Nz] = n_xz

x_eval = NNx//2
z_eval = 10 #NNz//2
plt.subplot(2, 2, 1)
plt.plot(np.real(myN[x_eval,:]))
plt.xlabel('z')
#plt.ylabel('I_0')
plt.title('r-cross-section of real(n)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(np.real(myN[:,z_eval]))
plt.xlabel('z')
#plt.ylabel('I_0')
plt.title('z-cross-section of real(n)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(np.imag(myN[x_eval,:]))
plt.xlabel('x')
#plt.ylabel('I_0')
plt.title('r-cross-section of imag(n)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(np.imag(myN[:,z_eval]))
plt.xlabel('x')
#plt.ylabel('I_0')
plt.title('z-cross-section of imag(n)')
plt.grid(True)

plt.tight_layout()
plt.show()


'''prepare for and carry out iteration:¶'''
k2_max = np.max(np.abs(myN**2-1)*k_0**2)
print("%.4f" % k2_max)
eps = 1.001 * k2_max  #Eq.(11)

# Fourier space and Green's function g0(kx,kz): 
kkx = np.fft.fftfreq(NNx) * 2*np.pi
kkz = np.fft.fftfreq(NNz) * 2*np.pi

KKx, KKz = np.meshgrid(kkx, kkz, sparse=True, indexing='ij')
g0 = 1./( np.square(KKx) + np.square(KKz) - k_0**2 - 1j*eps)
plt.imshow(np.fft.fftshift(g0.real))

# convolution of S with g0:
convS = np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(S)) *g0)


# scattering potential:
V = (myN**2 - 1)*k_0**2 - 1j*eps
V = np.fft.ifftshift(V)

gamma = 1j/eps * V


# initial field:
psi_0 = gamma * convS
psi = psi_0 
Upsi = gamma * (np.fft.ifftn(np.fft.fftn(psi_0*V)*g0)  - psi_0 + convS)
#print(psi.shape)

#%% Born series iteration:
# NOTE: DO NOT ITERATE BEYOND "CONVERGENCE" (as errors from the corners of the grid build up!)

N_iter = 1400
image_incr = 50
iw = 0
iter_accurr = 1.e-6 

while np.max(np.abs(Upsi)**2)  >= iter_accurr * np.max(np.abs(psi)**2) and iw <= N_iter-1:
    convPsi = np.fft.ifftn(np.fft.fftn(psi*V)*g0) 
    Upsi = gamma * (convPsi - psi + convS)
    psi = psi + Upsi  
    iw = iw+1

    if np.remainder(iw,image_incr) == 0:
        fig,(a0,a1)=plt.subplots(1,2,figsize=(12,8))
        im0 = a0.imshow(np.abs(np.fft.fftshift(Upsi))**2)
        a0.set_title('incremetal wave (Upsi), iteration = '+str(iw))
        a0.set_xlabel('z')
        a0.set_ylabel('x')
        plt.colorbar(im0,ax=a0,fraction=.045)
        im1 = a1.imshow(np.abs(np.fft.fftshift(psi))**2)
        a1.set_title('current psi, iteration = '+str(iw))
        a1.set_xlabel('z')
        a1.set_ylabel('x')
        plt.colorbar(im1,ax=a1,fraction=.045)
        np.max(np.abs(psi)**2) 
        plt.tight_layout()
        plt.show()
            
print('terminated at iteration = '+str(iw))
EE = np.fft.fftshift(psi)


E_iter = EE[n_pad:n_pad+N,n_pad:n_pad+Nz]
I_iter = np.abs(E_iter)**2
print(E_iter.shape)

z0 = 250
plt.subplot(2, 1, 1)
plt.plot(n_x/4,np.real(E_iter[:,z0]))
plt.xlabel('x[Lambda_0]')
#plt.ylabel('I_0')
plt.title('amplitude after iterating in x-space (central cross-section)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(n_x/4,I_iter[:,z0])
plt.xlabel('x[Lambda_0]')
#plt.ylabel('I_iter')
plt.title('intensity after iterating')
plt.grid(True)

plt.tight_layout()
plt.show()


img_result = plt.imshow(I_iter)
#img_object = plt.imshow(n_xz, cmap = 'Greys' , alpha=0.2)
img_object = plt.imshow(n_0*(1+dn)-n_xz, cmap = 'Greys' , alpha=0.2)
plt.show()