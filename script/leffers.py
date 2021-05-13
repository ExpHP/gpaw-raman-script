# coding: utf-8

#General
import numpy as np
from scipy import signal
from math import pi

# GPAW/ASE
from gpaw import GPAW, PW, FermiDirac
from gpaw.fd_operators import Gradient

from ase.phonons import Phonons
from ase.parallel import rank, size, world, MPI4PY, parprint

def get_elph_elements(atoms, gpw_name, calc_fd, sc = (1,1,1), basename = None):
    """
        Evaluates the dipole transition matrix elements

        Input
        ----------
        params_fd : Calculation parameters used for the phonon calculation
        sc (tuple): Supercell, default is (1,1,1) used for gamma phonons
        basename  : If you want give a specific name (gqklnn_{}.pckl)

        Output
        ----------
        gqklnn.pckl, the electron-phonon matrix elements
    """
    from ase.phonons import Phonons
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    par = MPI4PY()
    par.comm.Barrier()
    calc_gs = GPAW(gpw_name)

    #calc_fd = GPAW(**params_fd)
    calc_gs.initialize_positions(atoms)
    kpts = calc_gs.get_ibz_k_points()
    nk = len(kpts)
    gamma_kpt = [[0,0,0]]
    nbands = calc_gs.wfs.bd.nbands
    qpts = gamma_kpt

    calc_fd.get_potential_energy()  # XXX needed to initialize C_nM ??????

    #Phonon calculation, We'll read the forces from the elph.run function
    #This only looks at gamma point phonons
    ph = Phonons(atoms = atoms, name="phonons", supercell = sc)
    ph.read()
    frequencies, modes = ph.band_structure(qpts, modes=True)

    if rank == 0:
        print("Phonon frequencies are loaded.")

    #Find el-ph matrix in the LCAO basis
    elph = ElectronPhononCoupling(atoms, calc=None, supercell = sc)

    elph.set_lcao_calculator(calc_fd)
    elph.load_supercell_matrix(basis="dzp", dump=1)
    if rank == 0:
        print("Supercell matrix is loaded")

    #Find the bloch expansion coefficients
    c_kn = np.zeros((nk, nbands, calc_gs.wfs.setups.nao), dtype = complex)

    for k in range(len(kpts)):
        c_k = calc_gs.wfs.collect_array("C_nM",k,0)
        if rank == 0:
            c_kn[k] = c_k

    par.comm.Bcast(c_kn, root = 0)

    #And we finally find the electron-phonon coupling matrix elements!
    g_qklnn = elph.bloch_matrix(c_kn = c_kn, kpts = kpts, qpts = qpts, u_ql = modes)
    if rank == 0:
        print("Saving the elctron-phonon coupling matrix")
        if basename is None:
            np.save("gqklnn.npy", np.array(g_qklnn))
        else:
            np.save("gqklnn_{}.npy".format(basename), np.array(g_qklnn))



def get_dipole_transitions(calc, momname = None, basename = None):
    """
    Finds the dipole matrix elements:
    <\psi_n|\nabla|\psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m> where psi_n = u_n(r)*exp(ikr).

    Input:
        atoms           Relevant ASE atoms object
        momname         Suffix for the dipole transition file
        basename        Suffix used for the gs.gpw file

    Output:
        dip_vknm.npy    Array with dipole matrix elements
    """

    par = MPI4PY()  # FIXME: use a comm from gpaw
    #calc = atoms.calc

    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]

    wfs = {}

    parprint("Distributing wavefunctions.")

    if not calc.wfs.positions_set:
        calc.initialize_positions()
    for k in range(nk):
        #Collects the wavefunctions and the projections to rank 0. Periodic -> u_n(r)
        wf = np.array([calc.wfs.get_wave_function_array(i, k, 0, realspace = True, periodic=True) for i in range(n)], dtype = complex)
        P_nI = calc.wfs.collect_projections(k,0)

        #Distributes the information to rank k % size.
        if world.rank == 0:
            if k % world.size == world.rank:
                wfs[k] = wf,P_nI
            else:
                par.comm.Send(P_nI, dest = k % world.size, tag = nk+k)
                par.comm.Send(wf, dest = k % world.size, tag = k)
        else:
            if k % world.size == world.rank:
                nproj = sum(setup.ni for setup in calc.wfs.setups)
                if not calc.wfs.collinear:
                    nproj *= 2
                P_nI = np.empty((calc.wfs.bd.nbands, nproj), calc.wfs.dtype)
                shape = () if calc.wfs.collinear else(2,)
                wf = np.tile(calc.wfs.empty(shape, global_array = True, realspace = True), (n,1,1,1))

                par.comm.Recv(P_nI, source = 0, tag = nk + k)
                par.comm.Recv(wf, source = 0, tag = k)

                wfs[k] = wf,P_nI

    parprint("Evaluating dipole transition matrix elements.")

    dip_vknm = np.zeros((3, nk, n, n), dtype=complex)
    overlap_knm = np.zeros((nk,n,n),dtype = complex)

    nabla_v = [Gradient(calc.wfs.gd, v, 1.0, 4, complex).apply for v in range(3)]
    phases = np.ones((3, 2), dtype=complex)
    grad_nv = calc.wfs.gd.zeros((n,3),complex)

    for k, (wf, P_nI) in wfs.items():
        #Calculate <phit|nabla|phit> for the pseudo wavefunction
        for v in range(3):
            for i in range(n):
                nabla_v[v](wf[i],grad_nv[i,v], phases)

        dip_vknm[:,k] = np.transpose(calc.wfs.gd.integrate(wf, grad_nv),(2,0,1))

        overlap_knm[k] = [calc.wfs.gd.integrate(wf[i], wf) for i in range(n)]
        k_v = np.dot(calc.wfs.kd.ibzk_kc[k],calc.wfs.gd.icell_cv) * 2 * pi
        dip_vknm[:,k] += 1j*k_v[:,None,None]*overlap_knm[None,k,:,:]

        #The PAW corrections are added - see https://wiki.fysik.dtu.dk/gpaw/dev/documentation/tddft/dielectric_response.html#paw-terms
        I1 = 0
        #np.einsum is slow but very memory efficient.
        for a, setup in enumerate(calc.wfs.setups):
            I2 = I1 + setup.ni
            P_ni = P_nI[:, I1:I2]
            dip_vknm[:,k,:,:] += np.einsum('ni,ijv,mj->vnm', P_ni.conj(), setup.nabla_iiv, P_ni)
            I1 = I2

    world.sum(dip_vknm)

    if world.rank == 0:
        if momname is None:
            np.save('dip_vknm.npy', dip_vknm)
        else:
            np.save('dip_vknm_{}.npy'.format(momname), dip_vknm)

def L(w, gamma = 10/8065.544):
    #Lorentzian
    lor = 0.5*gamma/(pi*((w.real)**2+0.25*gamma**2))
    return lor

def calculate_raman_tensor(atoms, gpw_name, sc = (1,1,1), permutations = True, ramanname = None, momname = None, basename = None, w_l = 2.54066, gamma_l = 0.2):
    """
    Calculates the first order Raman tensor

    Input:
        atoms           ASE atoms object used for the phonon calculation
        sc              Supercell from the phonon calculation
        permutations    Used all fermi terms (True) or only the resonant term (False)
        ramanname       Suffix for the raman.npy file
        momname         Suffix for the momentumfile
        basename        Suffix for the gs.gpw and gqklnn.npy files
        w_l, gamma_l    Laser energy, broadening factor for the electron energies

    Output:
        RI.npy          Numpy array containing the raman spectre
    """

    par = MPI4PY()  # FIXME use a comm from gpaw

    parprint("Calculating the Raman spectra: Laser frequency = {}".format(w_l))

    calc = GPAW(gpw_name)  # FIXME: take calc as input

    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]
    cm = 1/8065.544
    ph = Phonons(atoms = atoms, name="phonons", supercell = sc)
    ph.read()
    w_ph= np.array(ph.band_structure([[0,0,0]])[0])
    w_cm = np.linspace(0, int(np.max(w_ph)/cm+200), int(np.max(w_ph)/cm+200)) #Defined in cm^-1
    w = w_cm*cm
    w_s = w_l-w
    m = len(w_ph)

    InvPh= np.array([])
    for l in range(m):
        if w_ph[l].real<0:
            InvPh = np.append(int(l), InvPh)

    k_info = {}

    if rank == 0:
        if momname is None:
            mom = np.load("dip_vknm.npy")[:,:,:,:] #[:,k,:,:]dim, k
        else:
            mom = np.load("dip_vknm_{}.npy".format(momname))[:,:,:,:] #[:,k,:,:]dim, k
        if basename is None:
            elph = np.load("gqklnn.npy")[0,:,:,:,:] #[0,k,l,:,:]
        else:
            elph = np.load("gqklnn_{}.npy".format(basename))[0,:,:,:,:] #[0,k,l,:,:]

    parprint("Distributing coupling terms")
    for k in range(nk):
        weight = calc.wfs.collect_auxiliary("weight", k, 0)
        f_n = calc.wfs.collect_occupations(k, 0)

        if world.rank == 0:
            f_n = f_n/weight
            if k % world.size == world.rank:
                #WEIGHTED
                k_info[k] = weight*elph[k], mom[:,k], f_n
            else:
                f_n = np.array(f_n, dtype = float)
                #WEIGHTED
                elph_k = weight*np.array(elph[k],dtype = complex)
                #elph_k = np.array(elph[k],dtype = complex)
                mom_k = np.array(mom[:,k],dtype = complex)

                par.comm.Send(elph_k, dest = k % world.size, tag = k)
                par.comm.Send(mom_k, dest = k % world.size, tag = nk + k)
                par.comm.Send(f_n, dest = k % world.size, tag = 2*nk + k)
        else:
            if k % world.size == world.rank:
                elph_k = np.empty((m,n,n), dtype = complex)
                mom_k = np.empty((3,n,n), dtype = complex)
                f_n = np.empty(n, dtype = float)

                par.comm.Recv(elph_k, source = 0, tag = k)
                par.comm.Recv(mom_k, source = 0, tag = nk + k)
                par.comm.Recv(f_n, source = 0, tag = 2*nk + k)

                k_info[k] = elph_k, mom_k, f_n

    #ab is in and out polarization
    #l is the phonon mode and w is the raman shift
    raman_ablw = np.zeros((3, 3, m, len(w)), dtype = complex)

    parprint("Evaluating Raman sum")

    E_kn = calc.band_structure().todict()["energies"][0]
    for k, (elph, mom, f_n) in k_info.items():
        print("For k = {}".format(k))
        E_el = E_kn[k]
        raman_ablw += np.einsum('asi,lij,bljs->abl', f_n[None,:,None]*(1-f_n[None,None, :])*mom/(w_l-(E_el[None,None,:]-E_el[None,:,None]) + complex(0,gamma_l)), elph, (1 - f_n[None,None,:,None])*mom[:,None, :,:]/(w_l-w_ph[None,:,None,None]-(E_el[None,None,:,None]- E_el[None,None,None,:]) + complex(0,gamma_l)))[:,:,:,None]

        # FIXME: GOOD GOD WHAT IS THIS
        if permutations:
            raman_ablw += np.einsum('asi,wljs,bij->ablw', f_n[None,:,None]*(1-f_n[None,None,:])*mom/(w_l-(E_el[None,None,:]-E_el[None,:,None])+ complex(0,gamma_l)), (1-f_n[None,None,:,None])*elph[None,:,:,:]/(w[:,None,None,None]-(E_el[None,None,:,None]-E_el[None,None,None,:])+ complex(0,gamma_l)),mom)

            raman_ablw += np.einsum('bwsi,lij,awljs->ablw',f_n[None,None,:,None]*(1-f_n[None,None,None,:])*mom[:,None,:,:]/(-w_s[None,:,None,None]-(E_el[None,None,None,:]-E_el[None,None, :, None])+ complex(0,gamma_l)), elph, (1-f_n[None,None,None,:,None])*mom[:,None,None,:,:]/(-w_s[None,:,None,None,None]-w_ph[None,None,:,None,None]-(E_el[None,None,None,:,None]-E_el[None,None,None, None,:])+ complex(0,gamma_l)))

            raman_ablw += np.einsum('bwsi,aij,wljs->ablw',f_n[None,None,:,None]*(1-f_n[None,None,None,:])*mom[:,None,:,:]/(-w_s[None,:,None,None]-(E_el[None,None,None,:]-E_el[None,None, :, None])+ complex(0,gamma_l)), mom, (1-f_n[None,None,:,None])*elph[None,:,:,:]/(-w_s[:,None,None,None]+w_l-(E_el[None,None,:,None]-E_el[None,None, None, :])+ complex(0,gamma_l)))

            raman_ablw += np.einsum('lsi,aij,bljs->abl', f_n[None,:,None]*(1-f_n[None,None,:])*elph/(-w_ph[:,None,None]-(E_el[None,None,:]-E_el[None,:,None])+ complex(0,gamma_l)), mom, (1-f_n[None,None,:,None])*mom[:,None,:,:]/(-w_ph[None,:,None,None]+w_l-(E_el[None,None,:,None]-E_el[None,None,None,:])+ complex(0,gamma_l)))[:,:,:,None]

            raman_ablw += np.einsum('lsi,bij,awljs->ablw', f_n[None,:,None]*(1-f_n[None,None,:])*elph/(-w_ph[:,None,None]-(E_el[None,None,:]-E_el[None,:,None])+ complex(0,gamma_l)),mom,(1-f_n[None,None,None,:,None])*mom[:,None,None,:,:]/(-w_ph[None,None,:,None,None]-w_s[None,:,None,None,None]-(E_el[None,None,None,:,None]-E_el[None,None,None,None,:])+ complex(0,gamma_l)))

    world.sum(raman_ablw)

    if rank == 0:
        if ramanname is None:
            np.save("Raman_tensor_ablw.npy", raman_ablw)
        else:
            np.save("Raman_tensor_{}_ablw.npy".format(ramanname), raman_ablw)

def calculate_raman(atoms, gpw_name, sc = (1,1,1), permutations = True, ramanname = None, momname = None, basename = None, w_l = 2.54066, gamma_l = 0.2, d_i = 0, d_o = 0):
    """
    Calculates the first order Raman spectre

    Input:
        atoms           ASE atoms object used for the phonon calculation
        sc              Supercell from the phonon calculation
        permutations    Used all fermi terms (True) or only the resonant term (False)
        ramanname       Suffix for the raman.npy file
        momname         Suffix for the momentumfile
        basename        Suffix for the gqklnn.npy files
        w_l, gamma_l    Laser energy, broadening factor for the electron energies
        d_i, d_o        Laser polarization in, out (0, 1, 2 for x, y, z respectively)
    Output:
        RI.npy          Numpy array containing the raman spectre
    """

    par = MPI4PY()  # FIXME: use a comm from gpaw

    parprint("Calculating the Raman spectra: Laser frequency = {}".format(w_l))

    calc = GPAW(gpw_name)  # FIXME: should take calc as arg instead

    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]
    cm = 1/8065.544

    ph = Phonons(atoms=atoms, name="phonons", supercell=sc)
    ph.read()
    w_ph= np.array(ph.band_structure([[0,0,0]])[0])
    w_cm = np.linspace(0, int(np.max(w_ph)/cm+200), int(np.max(w_ph)/cm+200)) #Defined in cm^-1
    w = w_cm*cm
    w_s = w_l-w
    m = len(w_ph)

    InvPh= np.array([])
    for l in range(m):
        if w_ph[l].real<0:
            InvPh = np.append(int(l), InvPh)

    k_info = {}

    if rank == 0:
        if momname is None:
            mom = np.load("dip_vknm.npy") #[:,k,:,:]dim, k
        else:
            mom = np.load("dip_vknm_{}.npy".format(momname)) #[:,k,:,:]dim, k
        if basename is None:
            elph = np.load("gqklnn.npy")[0] #[0,k,l,:,:]
        else:
            elph = np.load("gqklnn_{}.npy".format(basename))[0] #[0,k,l,n,m]

    parprint("Distributing coupling terms")
    for k in range(nk):
        weight = calc.wfs.collect_auxiliary("weight", k, 0)
        f_n = calc.wfs.collect_occupations(k, 0)

        if world.rank == 0:
            f_n = f_n/weight
            if k % world.size == world.rank:
                #WEIGHTED
                k_info[k] = weight*elph[k], mom[:,k], f_n
                #k_info[k] = elph[k], mom[:,k], f_n
            else:
                f_n = np.array(f_n, dtype = float)
                #WEIGHTED
                elph_k = weight*np.array(elph[k],dtype = complex)
                #elph_k = np.array(elph[k],dtype = complex)
                mom_k = np.array(mom[:,k],dtype = complex)

                par.comm.Send(elph_k, dest = k % world.size, tag = k)
                par.comm.Send(mom_k, dest = k % world.size, tag = nk + k)
                par.comm.Send(f_n, dest = k % world.size, tag = 2*nk + k)
        else:
            if k % world.size == world.rank:
                elph_k = np.empty((m,n,n), dtype = complex)
                mom_k = np.empty((3,n,n), dtype = complex)
                f_n = np.empty(n, dtype = float)

                par.comm.Recv(elph_k, source = 0, tag = k)
                par.comm.Recv(mom_k, source = 0, tag = nk + k)
                par.comm.Recv(f_n, source = 0, tag = 2*nk + k)

                k_info[k] = elph_k, mom_k, f_n

    #ab is in and out polarization
    #l is the phonon mode and w is the raman shift
    raman_lw = np.zeros((m, len(w)), dtype = complex)

    parprint("Evaluating Raman sum")

    E_kn = calc.band_structure().todict()["energies"][0]

    for k, (elph, mom, f_n) in k_info.items():
        print("For k = {}".format(k))
        E_el = E_kn[k]
        raman_lw += np.einsum('si,lij,ljs->l', f_n[:,None]*(1-f_n[None, :])*mom[d_i]/(w_l-(E_el[None,:]-E_el[:,None]) + complex(0,gamma_l)), elph, (1 - f_n[None,:,None])*mom[d_o,None, :,:]/(w_l-w_ph[:,None,None]-(E_el[None,:,None]- E_el[None,None,:]) + complex(0,gamma_l)))[:,None]

        # FIXME: GOOD GOD WHAT IS THIS
        if permutations:
            raman_lw += np.einsum('si,wljs,ij->lw', f_n[:,None]*(1-f_n[None,:])*mom[d_i]/(w_l-(E_el[None,:]-E_el[:,None])+ complex(0,gamma_l)), (1-f_n[None,None,:,None])*elph[None,:,:,:]/(w[:,None,None,None]-(E_el[None,None,:,None]-E_el[None,None,None,:])+ complex(0,gamma_l)),mom[d_o])

            raman_lw += np.einsum('wsi,lij,wljs->lw',f_n[None,:,None]*(1-f_n[None,None,:])*mom[d_o,None,:,:]/(-w_s[:,None,None]-(E_el[None,None,:]-E_el[None, :, None])+ complex(0,gamma_l)), elph, (1-f_n[None,None,:,None])*mom[d_i,None,None,:,:]/(-w_s[:,None,None,None]-w_ph[None,:,None,None]-(E_el[None,None,:,None]-E_el[None,None, None,:])+ complex(0,gamma_l)))

            raman_lw += np.einsum('wsi,ij,wljs->lw',f_n[None,:,None]*(1-f_n[None,None,:])*mom[d_o,None,:,:]/(-w_s[:,None,None]-(E_el[None,None,:]-E_el[None, :, None])+ complex(0,gamma_l)), mom[d_i], (1-f_n[None,None,:,None])*elph[None,:,:,:]/(-w_s[:,None,None,None]+w_l-(E_el[None,None,:,None]-E_el[None,None, None, :])+ complex(0,gamma_l)))

            raman_lw += np.einsum('lsi,ij,ljs->l', f_n[None,:,None]*(1-f_n[None,None,:])*elph/(-w_ph[:,None,None]-(E_el[None,None,:]-E_el[None,:,None])+ complex(0,gamma_l)), mom[d_i], (1-f_n[None,:,None])*mom[d_o,None,:,:]/(-w_ph[:,None,None]+w_l-(E_el[None,:,None]-E_el[None,None,:])+ complex(0,gamma_l)))[:,None]

            raman_lw += np.einsum('lsi,ij,wljs->lw', f_n[None,:,None]*(1-f_n[None,None,:])*elph/(-w_ph[:,None,None]-(E_el[None,None,:]-E_el[None,:,None])+ complex(0,gamma_l)),mom[d_o],(1-f_n[None,None,:,None])*mom[d_i,None,None,:,:]/(-w_ph[None,:,None,None]-w_s[:,None,None,None]-(E_el[None,None,:,None]-E_el[None,None,None,:])+ complex(0,gamma_l)))

    world.sum(raman_lw)

    RI = np.zeros(len(w))
    for l in range(m):
        if not (l in InvPh):
            parprint("Phonon {} with energy = {} registered".format(l, w_ph[l]))
            RI += (np.abs(raman_lw[l])**2)*np.array(L(w-w_ph[l]))

    raman = np.vstack((w_cm, RI))

    if rank == 0:
        if ramanname is None:
            np.save("RI.npy", raman)
        else:
            np.save("RI_{}.npy".format(ramanname), raman)

def analyse_raman_tensor(atoms,basename =  None, sc = (1,1,1), ramanname = None, tensorname = None, d_i = [1,0,0], d_o = [1,0,0], w_ph = None):
    """
        Evaluates the Raman spectrum for a given polarisation

        Input:
            basename  : Suffix for the gs.gpw and gqklnn.npy files
            sc        : Supercell used for the phonon calculation
            ramanname : Suffix for the evaluated Raman spectrum, RI.npy
            tensorname: Suffix for the Raman tensor file, Raman_tensor_ablw.npy
            d_i, d_o  : The polarisation of the incoming and outgoing photons

        Output:
            RI.npy    : Evaluated Raman spectrum
    """

    cm = 1/8065.544
    m = len(w_ph)

    d_in = np.array(d_i)
    d_in = d_in/np.linalg.norm(d_in)

    d_out = np.array(d_o)
    d_out = d_out/np.linalg.norm(d_out)


    if tensorname is None:
        raman_ablw = np.load("Raman_tensor_ablw.npy")
    else:
        raman_ablw = np.load("Raman_tensor_{}_ablw.npy".format(tensorname))

    InvPh= np.array([])
    for l in range(m):
        if w_ph[l].real<0:
            InvPh = np.append(int(l), InvPh)
    w_cm = np.linspace(0, int(np.max(w_ph)/cm+200), int(np.max(w_ph)/cm+200)) #Defined in cm^-1
    w = w_cm*cm

    RI = np.zeros(len(w))
    for l in range(m):
        if not (l in InvPh):
            RI += (np.abs(np.einsum('a,ablw,b->lw',d_in, raman_ablw, d_out)[l])**2)*np.array(L(w-w_ph[l]))

    raman = np.vstack((w_cm, RI))

    if rank == 0:
        if ramanname is None:
            np.save("RI.npy", raman)
        else:
            np.save("RI_{}.npy".format(ramanname), raman)

def plot_raman(yscale = "linear", figname = "Raman.png", relative = False, w_min = None, w_max = None, ramanname = None):
    """
        Plots a given Raman spectrum

        Input:
            yscale: Linear or logarithmic yscale
            figname: Name of the generated figure
            relative: Scale to the highest peak
            w_min, w_max: The plotting range wrt the Raman shift
            ramanname: Suffix used for the file containing the Raman spectrum

        Output:
            ramanname: image containing the Raman spectrum.

    """
    import matplotlib
    matplotlib.use('Agg')  # FIXME: Evil, none of this function's business
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    #Plotting function

    if rank == 0:
        if ramanname is None:
            legend = False
            RI_name = ["RI.npy"]
        elif type(ramanname) == list:
            legend = True
            RI_name = ["RI_{}.npy".format(name) for name in ramanname]
        else:
            legend = False
            RI_name = ["RI_{}.npy".format(ramanname)]

        ylabel = "Intensity (arb. units)"
        inferno = cm = plt.get_cmap('inferno')
        cNorm  = colors.Normalize(vmin=0, vmax=len(RI_name))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        peaks = None
        for i, name in enumerate(RI_name):
            RI = np.real(np.load(name))
            if w_min == None:
                w_min = np.min(RI[0])
            if w_max == None:
                w_max = np.max(RI[0])
            r = RI[1][np.logical_and(RI[0]>=w_min, RI[0]<=w_max)]
            w = RI[0][np.logical_and(RI[0]>=w_min, RI[0]<=w_max)]
            cval = scalarMap.to_rgba(i)
            if relative:
                ylabel = "I/I_max"
                r = r/np.max(r)
            if peaks is None:
                peaks = signal.find_peaks(r[np.logical_and(w>=w_min, w<=w_max)])[0]
                locations = np.take(w[np.logical_and(w>=w_min, w<=w_max)], peaks)
                intensities = np.take(r[np.logical_and(w>=w_min, w<=w_max)], peaks)
            if legend:
                plt.plot(w, r, color = cval, label = ramanname[i])
            else:
                plt.plot(w, r, color = cval)
        for i, loc in enumerate(locations):
            if intensities[i]/np.max(intensities)>0.05:
                plt.axvline(x = loc,  color = "grey", linestyle = "--")

        # FIXME: usage of pyplot API
        plt.yscale(yscale)
        plt.minorticks_on()
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Raman intensity")
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel(ylabel)
        if not relative:
            plt.yticks([])
        plt.savefig(figname, dpi=300)
        plt.clf()
