import constants as const
import numpy as np
'''
MIT License

Copyright (c) 2023 Andreas Ekström, Chalmers University of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

class nn_studio:

    def __init__(self,jmin,jmax,tzmin,tzmax,Np=75,mesh_type='gauleg_infinite'):

        self.jmin = jmin
        self.jmax = jmax
        self.tzmin = tzmin
        self.tzmax = tzmax
        
        self.basis = self.setup_NN_basis()
        self.channels = self.setup_NN_channels()
        
        self.Np = Np
        if mesh_type == 'gauleg_infinite':
            self.pmesh, self.wmesh = self.gauss_legendre_inf_mesh()
        elif mesh_type == 'gauleg_finite':
            self.pmesh, self.wmesh = self.gauss_legendre_line_mesh(1e-16,1000)
            
        self.lecs = None

        self.Tlabs = None
        
        #potential
        self.V = None

        #Tmatrices
        self.Tmtx = []

        #phase shifts
        self.phase_shifts = []
        
    def setup_NN_basis(self):
        basis = []
        for tz in range(self.tzmin,self.tzmax+1,1):
            for J in range(self.jmin,self.jmax+1,1):
                for S in range(0,2,1):
                    for L in range(abs(J-S),J+S+1,1):
                        for T in range(abs(tz),2,1):
                            if ((L+S+T)%2 != 0):
                                basis_state = {}
                                basis_state['tz'] = tz
                                basis_state['l']  = L
                                basis_state['pi'] = (-1)**L
                                basis_state['s']  = S
                                basis_state['j']  = J
                                basis_state['t']  = T
                                basis.append(basis_state)
        return basis

    def setup_NN_channels(self):

        from itertools import groupby
        from operator import itemgetter
        
        states = []
        
        for bra in self.basis:
            for ket in self.basis:
            
                if self.kroenecker_delta(bra,ket,'j','tz','s','pi'):
                
                    state = {}
                    
                    state['l']  = bra['l']
                    state['ll'] = ket['l']
                    
                    state['s']  = bra['s']
                    state['j']  = bra['j']
                    state['t']  = bra['t']
                    state['tz'] = bra['tz']
                    state['pi'] = bra['pi']
                    states.append(state)
                    
        grouper = itemgetter("s", "j", "tz", "pi")
        NN_channels = []
    
        for key, grp in groupby(sorted(states, key = grouper), grouper):
            NN_channels.append(list(grp))
        

        for chn_idx, chn in enumerate(NN_channels):
            for block in chn:
                block.update({"chn_idx":chn_idx})

        return NN_channels

    def lookup_channel_idx(self, **kwargs):
        matching_indices = []
        channels = []
        for idx, chn in enumerate(self.channels):
            for block in chn:
                if (kwargs.items() <= block.items()):
                    matching_indices.append(idx)
                    channels.append(chn)

        matching_indices = list(dict.fromkeys(matching_indices))

        return matching_indices, channels

    def linear_mesh(self):

        return np.linspace(1e-6,650,self.Np)
    
    def gauss_legendre_line_mesh(self,a,b):
        x, w = np.polynomial.legendre.leggauss(self.Np)
        # Translate x values from the interval [-1, 1] to [a, b]
        t = 0.5*(x + 1)*(b - a) + a
        u = w * 0.5*(b - a)

        return t,u
    
    def gauss_legendre_inf_mesh(self):

        scale=100.0
        
        x, w = np.polynomial.legendre.leggauss(self.Np)
        
        # Translate x values from the interval [-1, 1] to [0, inf)
        pi_over_4 = np.pi/4.0
        
        t = scale*np.tan(pi_over_4*(x+1.0))
        u = scale*pi_over_4/np.cos(pi_over_4*(x+1.0))**2*w
        
        return t,u

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def triag(a, b, ab):
        if( ab < abs(a - b) ):
            return False
        if ( ab > a + b ):
            return False
        return True
    
    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def kroenecker_delta(bra,ket,*args):
        for ar in args:
            if bra[ar] != ket[ar]:
                return False
        return True

    def lab2rel(self,Tlab,tz):
    
        if tz == -1:
            mu = const.Mp/2
            ko2 = 2*const.Mp*Tlab
        elif tz ==  0:
            mu = const.Mp*const.Mn/(const.Mp+const.Mn)
            ko2 = const.Mp**2*Tlab*(Tlab+2*const.Mn)/((const.Mp+const.Mn)**2 + 2*Tlab*const.Mp)
        elif tz == +1:
            mu = const.mN/2
            ko2 = 2*const.Mp*Tlab
        else:
            exit('unknown isospin projection')

        if ko2<0:
            ko = np.complex(0,np.sqrt(np.abs(ko2)))
        else:    
            ko = np.sqrt(ko2)

        return ko,mu

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def map_to_coup_idx(ll,l,s,j):

        if l == ll:
        
            if l<j:
                # --
                coup = True
                idx  = 3
            elif l>j:
                # ++
                coup = True
                idx  = 2
            else:
                if s==1:
                    coup = False
                    idx  = 1
                else:
                    coup = False
                    idx  = 0
        else:
            if l<j:
                # -+
                coup = True
                idx  = 5
            else:
                # +-
                coup = True
                idx  = 4

        return coup,idx
    
    def Vmtx(self,this_mesh,ll,l,s,j,t,tz):
    
        coup,idx = self.map_to_coup_idx(ll,l,s,j)
        mtx = np.zeros((len(this_mesh), len(this_mesh)))
        for pidx, p in enumerate(this_mesh):
            for ppidx, pp in enumerate(this_mesh):
                mtx[ppidx][pidx] = self.V.potential(pp,p,coup,s,j,t,tz,self.lecs)[idx]
        return np.array(mtx)
    
    def setup_Vmtx(self,this_channel,ko=False):

        if ko==False:
            this_mesh = self.pmesh
        else:
            this_mesh = np.hstack((self.pmesh,ko))
        m = []

        for idx, block in enumerate(this_channel):
            
            l  = block['l']
            ll = block['ll']
            s  = block['s']
            j  = block['j']
            t  = block['t']
            tz = block['tz']
                
            mtx = np.copy(self.Vmtx(this_mesh,ll,l,s,j,t,tz))

            m.append(mtx)

        if len(this_channel) >1:
            V = np.copy(np.vstack((np.hstack((m[0],m[1])),
                                   np.hstack((m[2],m[3])))))
        else:
            V = np.copy(m[0])
                   
        return V, m

    def setup_G0_vector(self,ko,mu):
            
        G = np.zeros((2*self.Np+2), dtype=complex)

        # note that we index from zero, and the N+1 point is at self.Np
        G[0:self.Np] = self.wmesh*self.pmesh**2/(ko**2 - self.pmesh**2)		# Gaussian integral

        #print('   G0 pole subtraction')
        G[self.Np]  = -np.sum( self.wmesh/(ko**2 - self.pmesh**2 ) )*ko**2 	# 'Principal value'
        G[self.Np] -= 1j*ko * (np.pi/2)

        #python vec[0:n] is the first n elements, i.e., 0,1,2,3,...,n-1
        G[self.Np+1:2*self.Np+2] = G[0:self.Np+1]
        return G*2*mu
    
    def setup_GV_kernel(self,channel,Vmtx,ko,mu):
    
        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko,mu)
        
        g = np.copy(G0[0:Np_chn])
        GV = np.zeros((len(g),len(g)),dtype=complex)
        
        for g_idx, g_elem in enumerate(g):
            GV[g_idx,:] = g_elem * Vmtx[g_idx,:]
            
        return GV

    def setup_VG_kernel(self,channel,Vmtx,ko,mu):

        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        
        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko,mu)
        g = np.copy(G0[0:Np_chn])
        VG = np.zeros((len(g),len(g)),dtype=complex)
        
        for g_idx, g_elem in enumerate(g):
            VG[:,g_idx] = g_elem * Vmtx[:,g_idx]
        
        return VG
    
    def solve_lippmann_schwinger(self,channel,Vmtx,ko,mu):

        # matrix inversion:
        # T = V + VGT
        # (1-VG)T = V
        # T = (1-VG)^{-1}V
        
        VG = self.setup_VG_kernel(channel,Vmtx,ko,mu)
        VG = np.eye(VG.shape[0]) - VG
        # golden rule of linear algebra: avoid matrix inversion if you can
        #T = np.matmul(np.linalg.inv(VG),Vmtx)
        T = np.linalg.solve(VG,Vmtx)

        return T

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def compute_phase_shifts(ko,mu,on_shell_T):

        rad2deg = 180.0/np.pi
        
        fac  = np.pi*mu*ko
        
        if len(on_shell_T) == 3:
            
            T11 = on_shell_T[0]
            T12 = on_shell_T[1]
            T22 = on_shell_T[2]
        
            # Blatt-Biedenharn (BB) convention
            twoEpsilonJ_BB = np.arctan(2*T12/(T11-T22))	# mixing parameter
            delta_plus_BB  = -0.5*1j*np.log(1 - 1j*fac*(T11+T22) + 1j*fac*(2*T12)/np.sin(twoEpsilonJ_BB))
            delta_minus_BB = -0.5*1j*np.log(1 - 1j*fac*(T11+T22) - 1j*fac*(2*T12)/np.sin(twoEpsilonJ_BB))

            # this version has a numerical instability that I should fix.
            # Stapp convention (bar-phase shifts) in terms of Blatt-Biedenharn convention
            #twoEpsilonJ = np.arcsin(np.sin(twoEpsilonJ_BB)*np.sin(delta_minus_BB - delta_plus_BB))      # mixing parameter
            #delta_minus = 0.5*(delta_plus_BB + delta_minus_BB + np.arcsin(np.tan(twoEpsilonJ)/np.tan(twoEpsilonJ_BB)))
            #delta_plus  = 0.5*(delta_plus_BB + delta_minus_BB - np.arcsin(np.tan(twoEpsilonJ)/np.tan(twoEpsilonJ_BB)))
            #epsilon     = 0.5*twoEpsilonJ

            # numerially stable conversion
            cos2e = np.cos(twoEpsilonJ_BB/2)*np.cos(twoEpsilonJ_BB/2)
            cos_2dp = np.cos(2*delta_plus_BB)
            cos_2dm = np.cos(2*delta_minus_BB)
            sin_2dp = np.sin(2*delta_plus_BB)
            sin_2dm = np.sin(2*delta_minus_BB)
            
            aR = np.real(cos2e*cos_2dm + (1-cos2e)*cos_2dp)
            aI = np.real(cos2e*sin_2dm + (1-cos2e)*sin_2dp)
            delta_minus = 0.5*np.arctan2(aI,aR)

            aR = np.real(cos2e*cos_2dp + (1-cos2e)*cos_2dm)
            aI = np.real(cos2e*sin_2dp + (1-cos2e)*sin_2dm)
            delta_plus = 0.5*np.arctan2(aI,aR)

            tmp = 0.5*np.sin(twoEpsilonJ_BB)
            aR = tmp*(cos_2dm - cos_2dp)
            aI = tmp*(sin_2dm - sin_2dp)
            tmp = delta_plus + delta_minus
            epsilon = 0.5*np.arcsin(aI*np.cos(tmp) - aR*np.sin(tmp)) 
            
            if ko <150:
                if delta_minus*rad2deg<0:
                    delta_minus += np.pi
                    epsilon *= -1.0
            return [np.real(delta_minus*rad2deg), np.real(delta_plus*rad2deg), np.real(epsilon*rad2deg)]
        
        else:
            # uncoupled
            T = on_shell_T[0]
            Z = 1-fac*2j*T
            # S=exp(2i*delta)
            delta = (-0.5*1j)*np.log(Z)

            return np.real(delta*rad2deg)
   
    def compute_Tmtx(self,channels,verbose=False):

        if verbose:
            print(f'computing T-matrices for')

        self.Tmtx = []
        self.phase_shifts = []

        for idx, channel in enumerate(channels):
            if verbose:
                print(f'channel = {channel}')

            phase_shifts_for_this_channel = []

            nof_blocks = len(channel)
                            
            for Tlab in self.Tlabs:

                if verbose:
                    print(f'Tlab = {Tlab} MeV')

                ko,mu= self.lab2rel(Tlab,channel[0]['tz'])
                Vmtx = self.setup_Vmtx(channel,ko)[0] # get only V, not the list of submatrices
                this_T = self.solve_lippmann_schwinger(channel,Vmtx,ko,mu)
                self.Tmtx.append(this_T)

                Np = this_T.shape[0]
                # extract the on-shell T elements
                if nof_blocks > 1:
                    #coupled
                    Np = int((Np-2)/2)
                    T11 = this_T[Np,Np]
                    T12 = this_T[2*Np+1,Np]
                    T22 = this_T[2*Np+1,2*Np+1]
                    on_shell_T = [T11,T12,T22]
                else:
                    # uncoupled
                    Np = Np-1
                    T11 = this_T[Np,Np]
                    on_shell_T = [T11]

                this_phase_shift = self.compute_phase_shifts(ko,mu,on_shell_T)
                phase_shifts_for_this_channel.append(this_phase_shift)

            self.phase_shifts.append(np.array(phase_shifts_for_this_channel))      

    def get_Vmtx_from_split(self,V_split,lec_vector):

        V = 0
        for idx, this_V in enumerate(V_split):

            V += lec_vector[idx]*this_V

        return V
            
    def model_order_reduction(self,channel,Tlab,directions,training_points,verbose=False):

        if len(channel) > 1:
            exit('model order reduction: limited to one channel at the time')

        if len(directions) != len(training_points[0]):
            exit('model order reduction: need training in every direction')
            
        nof_blocks = len(channel[0])
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        Np = Np_chn-1
        if nof_blocks == 4:
            Np = int((Np_chn-2)/2)
                    
        if verbose:
            print(f'MOR:')
            print(f'channel          = {channel}')
            print(f'nof_blocks       = {nof_blocks}')
            print(f'Np_chn           = {Np_chn}')
            print(f'Tlab             = {Tlab}')
            print(f'directions       = {directions}')
            print(f'#training points = {len(training_points)}')
            print(f'{training_points}')

        # we use nof_blocks to identify coupled (=4) and uncoupled (=1) channels
                
        ko,mu= self.lab2rel(Tlab,channel[0][0]['tz'])
        
        # we construct a diagonal matrix from the vector G0
        G = np.diag(self.setup_G0_vector(ko,mu)[0:Np_chn])
        
        # set relevant lecs to zero to extract constant part
        for lec, value in self.lecs.items():
            if lec in directions:
                self.lecs[lec] = 0
                
        V_split = []
        V_split.append(self.setup_Vmtx(channel[0],ko)[0]) # get only V, not the list of submatrices

        # set lecs in each direction to 1.0 and get the split potential term
        for lec, value in self.lecs.items():
            if lec in directions:
                self.lecs[lec] = 1.0
                V_direction = self.setup_Vmtx(channel[0],ko)[0] # get only V, not the list of submatrices
                V_split.append(V_direction - V_split[0])
                self.lecs[lec] = 0.0
                    
        # we need GV, VG for each split V term
        GV_split = []
        VG_split = []
        GVG_split = []
        Vi_split_T11 = [] # the on-shell Vi-split terms
        Vi_split_T12 = [] # the on-shell Vi-split terms
        Vi_split_T22 = [] # the on-shell Vi-split terms
        for this_V in V_split:
            GV = self.setup_GV_kernel(channel[0],this_V,ko,mu)
            VG = self.setup_VG_kernel(channel[0],this_V,ko,mu)
            Vi_split_T11.append(this_V[Np,Np])
            if (nof_blocks == 4):
                Vi_split_T12.append(this_V[2*Np+1,Np])
                Vi_split_T22.append(this_V[2*Np+1,2*Np+1])
            GV_split.append(GV)
            VG_split.append(VG)
            GVG_split.append(GV@G)
                
        # loop over the training lec values and setup the relevant lec vector
        # to assemble the potential for the split terms
        Ti = []
        for training_point in training_points:
            lec_training_vector = []
            lec_training_vector.append(1.0)
            for lec, value in self.lecs.items():
                if lec in directions:
                    # training points and directions are identically ordered wrt lec names
                    lec_training_vector.append(training_point[directions.index(lec)])

            V = self.get_Vmtx_from_split(V_split,lec_training_vector)
            this_T = self.solve_lippmann_schwinger(channel[0],V,ko,mu)
            Ti.append(self.solve_lippmann_schwinger(channel[0],V,ko,mu))

        # construct the emulator
        mi_split_T11  = np.zeros((len(training_points)), dtype=object)
        Mij_split_T11 = np.zeros((len(training_points),len(training_points)), dtype=object)
        Mij_const_T11 = np.zeros((len(training_points),len(training_points)), dtype=object)

        if (nof_blocks == 4):
            mi_split_T12  = np.zeros((len(training_points)), dtype=object)
            mi_split_T22  = np.zeros((len(training_points)), dtype=object)
            Mij_split_T12 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_split_T22 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_const_T12 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_const_T22 = np.zeros((len(training_points),len(training_points)), dtype=object)
            
        for r,training_point in enumerate(training_points):
            mi_split_T11[r] = []
            if (nof_blocks == 4):
                mi_split_T12[r] = []
                mi_split_T22[r] = []
            for c,training_point in enumerate(training_points):
                Mij_split_T11[r,c] = []
                Mij_const_T11[r,c] = []
                if (nof_blocks == 4):
                    Mij_split_T12[r,c] = []
                    Mij_const_T12[r,c] = []
                    Mij_split_T22[r,c] = []
                    Mij_const_T22[r,c] = []
                    
        for split_idx, _ in enumerate(GV_split):
            for r,training_point in enumerate(training_points):
                this_mi_TGV = np.copy(Ti[r]@GV_split[split_idx])
                this_mi_VGT = np.copy(VG_split[split_idx]@Ti[r])
                mi_split_T11[r].append((this_mi_TGV + this_mi_VGT)[Np,Np])
                if (nof_blocks == 4):
                    mi_split_T12[r].append((this_mi_TGV + this_mi_VGT)[2*Np+1,Np])
                    mi_split_T22[r].append((this_mi_TGV + this_mi_VGT)[2*Np+1,2*Np+1])
                for c,training_point in enumerate(training_points):
                    this_TGVGT_rc = np.copy(-Ti[r]@GVG_split[split_idx]@Ti[c])
                    this_TGVGT_cr = np.copy(-Ti[c]@GVG_split[split_idx]@Ti[r])
                    Mij_split_T11[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[Np,Np])
                    Mij_const_T11[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[Np,Np]
                    if (nof_blocks == 4):
                        Mij_split_T12[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[2*Np+1,Np])
                        Mij_const_T12[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[2*Np+1,Np]
                        Mij_split_T22[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[2*Np+1,2*Np+1])
                        Mij_const_T22[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[2*Np+1,2*Np+1]
                       
        def emulator(emulator_lecs,verbose=False):

            ntp = len(training_points)
            this_ko   = ko
            this_mu   = mu
            this_Tlab = Tlab
            this_channel = channel
            this_nof_blocks = nof_blocks
            
            if verbose:
                print(f'emulating for lecs = {emulator_lecs}')
            
            # construct m and M
            nof_split_terms = len(mi_split_T11[0])
            nof_emulator_lecs = len(emulator_lecs)
            
            m_i_T11 = np.zeros((ntp), dtype=complex)
            M_ij_T11 = np.zeros((ntp,ntp), dtype=complex)
            V_T11 = 0
            if this_nof_blocks==4:
                m_i_T12 = np.zeros((ntp), dtype=complex)
                M_ij_T12 = np.zeros((ntp,ntp), dtype=complex)
                V_T12 = 0
                m_i_T22 = np.zeros((ntp), dtype=complex)
                M_ij_T22 = np.zeros((ntp,ntp), dtype=complex)
                V_T22 = 0
                
            assert nof_split_terms == nof_emulator_lecs , f"{nof_split_terms} /= {nof_emulator_lecs}"
            for r in range(0,ntp):
                for idx,lec in enumerate(emulator_lecs):
                    m_i_T11[r]  += lec*mi_split_T11[r][idx]
                    if this_nof_blocks == 4:
                        m_i_T12[r]  += lec*mi_split_T12[r][idx]
                        m_i_T22[r]  += lec*mi_split_T22[r][idx]
                for c in range(0,ntp):
                    for idx,lec in enumerate(emulator_lecs):
                        M_ij_T11[r,c] += lec*Mij_split_T11[r,c][idx]
                        if this_nof_blocks == 4:
                            M_ij_T12[r,c] += lec*Mij_split_T12[r,c][idx]
                            M_ij_T22[r,c] += lec*Mij_split_T22[r,c][idx]
                            
                    M_ij_T11[r,c] += Mij_const_T11[r,c]
                    if this_nof_blocks == 4:
                        M_ij_T12[r,c] += Mij_const_T12[r,c]
                        M_ij_T22[r,c] += Mij_const_T22[r,c]
                        
            for idx,lec in enumerate(emulator_lecs):
                V_T11 += lec*Vi_split_T11[idx]
                if this_nof_blocks == 4:
                    V_T12 += lec*Vi_split_T12[idx]
                    V_T22 += lec*Vi_split_T22[idx]
                    
            T11 = V_T11 + 0.5*m_i_T11.T@np.linalg.inv(M_ij_T11)@m_i_T11
            if this_nof_blocks == 4:
                T12 = V_T12 + 0.5*m_i_T12.T@np.linalg.inv(M_ij_T12)@m_i_T12
                T22 = V_T22 + 0.5*m_i_T22.T@np.linalg.inv(M_ij_T22)@m_i_T22

            if this_nof_blocks == 4:
                return [T11,T12,T22],this_Tlab,this_ko,this_mu
            return [T11],this_Tlab,this_ko,this_mu

        return emulator
