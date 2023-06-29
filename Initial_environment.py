from neurodevsim.simulator import *
import numpy as np

from sympy.solvers import solve
from sympy import Point3D, Plane, symbols

import math
from heapq import nlargest

import time


# mode 0: find target nearby
# mode 1: tangential migration to the target
# mode 1.5 --> 2: close enough to extend root_axon and start radial migration
# mode 2 --> 3: (after extended root axon) radially migrating
# mode 3 --> 4: leave BG proc guidance [expected to start experiencing interference by BG and PC somata]
# mode 4 --> 5: arrived at IGL_z
# mode -1: stuck

redirect = False # for using BG detection

verbose_cell = empty_ID#ID(3,588)
verbose = 1
# 1: dendritic selection
# 2: synapses
# 3: rare cases
# 4: morphology
# 5: Bergmann Glia
# 6: Failed Granule Cells

byFronts = False # False = retract by signal input


PCL_z = 4 # top of PCL in average (top line of PC soma), IGL_z in nds_analyze_cerebellum

# global variables for PCs
PC_start_growth = 65
wholeCheck_cycle_1 = PC_start_growth + 14 # = 79
wholeCheck_cycle_2 = wholeCheck_cycle_1 + 7 # = 86

wholeCheck_1 = 200 # threshold of front numbers in a whole neuron to start first retraction process
wholeCheck_2 = 250

first_f_th = 20
# no 'second_f_th', fixed to choose one with maximum numbers of fronts

PC_root_radius = 0.05
PC_root_offsets_y = [-6.5, -3.5, -0.5, 3, 6]
PC_root_num = 5
PC_root_len = 7.5

PC_growth_step = 5.0
PC_taper = 1.0

PC_branch_taper = 1.0
degree = 65 # mean bifurcation angle 65 degree in vivo (Fujishima et al.,2012),

repel_fac = 0.6
repel_lim = 1.0

# global variables for BGs #
BG_offsets = [-20.5, -10.,-4.75,-0.5,4.75,10.,19.5]
num_roots = 7
BGproc_radi = 0.5
root_len = 4#

proc_len = 4
pia = 0 + 6 + 125 # PC soma position + PC soma radi + ML in P11 mice (Komuro and Rakic 1995)

# global variables for GCs #
axon_root_len = 2.0
axon_root_x_coord = 5.0

initial_pf_step = Point(0,2.,0)
pf_step = Point(0,4.,0)
radial_step = Point(0,0,-5)
ymax = 119.9
ymin = -19.9
IGL_z = -15

f_radi = 0.1 # sync with soma_radi
BG_GCoffset = (BGproc_radi + f_radi) * 1.1


close_but_closer = 0.3
ag_step = Point(0,0,-4)

syn_distance = 1.0

max_loop = 3

# Signal prams (from Synapses.ipynb example)
MAX_FR = 0.2
NEW_FR = 10.0
SIGNAL_DECAY = 20.


class PurkinjeCell(SynFront):
    # status1(): False = first solve_collision_PC_den trial, Ture = second and the last trial
    
    _fields_ = Front._fields_ + [('prev_branch', c_short),('signal', c_double),('stage', c_short)]
    
    def manage_front(self,constellation):
    
        if constellation.cycle == 1: # initially Purkinje cells are not active
            self.disable(constellation,till_cycle_g=PC_start_growth) # re-activate at PC_start_growth

        ID = self.get_id()
        somaID = self.get_soma(constellation,returnID=True)
        
        if self.swc_type == 1:
            roots = self.get_children(constellation)
            if constellation.cycle == PC_start_growth:
                self.stage = 1
                self.make_dendrite_roots(constellation,somaID)
            elif constellation.cycle >= wholeCheck_cycle_1 and self.stage == 1:
                print ("ERROR! in PC manage_front: PC reached wholeCheck_cycle_1 =",wholeCheck_cycle_1)
            else:
                return
        elif (self.swc_type == 5) or (self.swc_type == 3):
            if self.has_synapse(): # front has a synapse and integrates syn_input each cycle
                self.signal += self.syn_input - (self.signal / SIGNAL_DECAY)
        
            # long enough -> stop growth
            if self.end.z >= 100: # ML thickness at P10 = ˜75µm (Fig1, Yamanaka, Yanagawa, Obata 2004)
                if (verbose >= 4) or (verbose_cell == somaID):
                        print (somaID,ID," : _extend_basal_dendrite_, long enough, disable at",self.end.z)
                self.disable(constellation)
        
            elif self.num_children == 0:
                self.extend_basal_dendrite(constellation,ID,somaID)
            else:
                self.disable(constellation)
                
            return
                
        else:
            print ("Error in manage_front PurkinjeCell: unexpected swc_type =",self.swc_type)



    def make_dendrite_roots(self,constellation,somaID):
        n_roots = 0
        for i in range(0,PC_root_num):
            x = np.random.uniform(-PC_root_len,PC_root_len)
            y = PC_root_offsets_y[i] + np.random.uniform(-1,1) # adding jitter on the offsets
            z = np.random.random()+ PC_root_len
            
            x_points = np.arange(-PC_root_len,PC_root_len+0.5,0.5)
            np.random.shuffle(x_points)
            new_name = self.get_neuron_name(constellation) + "root" + str(i)
            for x_point in x_points:
                direction = Point(x_point,y,z)
                norm_dir = direction.norm() # turn the direction to unit vector
                new_pos = self.orig + norm_dir * PC_root_len
                try:
                    new_front = self.add_child(constellation,new_pos,\
                                            radius=PC_root_radius,\
                                            branch_name=new_name,swc_type=5)
                    n_roots += 1
                    break # success
                except (CollisionError, GridCompetitionError, InsideParentError, VolumeError):
                    continue # pick another point, no attempt to correct the error
        # print error messages if less roots made
        if n_roots == 0:
            print (somaID,": _make_dendrite_root_ no valid root dendrite found")
        elif n_roots < PC_root_num:
            print (somaID,"_make_dendrite_root_ less roots",n_roots)

    def extend_basal_dendrite(self,constellation,ID,somaID):
       

        if (self.prev_branch == 0) and (np.random.random() < 0.08): # branch rate in vitro 0.26times/3hrs (Fujishima et al.,2012)
            self.branching(constellation,ID,somaID)
            return

        #search front for adding repulsion force
        other_dendrites = self.get_fronts(constellation,what='other',max_distance=10,swc_types=[3,5])
        sorted_other_dendrites = sorted(other_dendrites, key=lambda e:abs(e[0].end.y-self.end.y))  # re-arrange order by distance in y-directions
        
        # get list of fronts from same cell but different branches
        samecell_dendrites = self.get_fronts(constellation,what='self',max_distance=10,swc_types=[3,5])
        self_branch_name = self.get_branch_name() # ex. 'PC_10_root0_'
        b = self_branch_name.find("root")
        self_root_name = self_branch_name[:b+5]#self_branch_name[:5] # ex. 'root1'
        del_candidates = [] # empty list to save items to be removed in the 2nd loop
        for i in range(len(samecell_dendrites)):
            den_front = samecell_dendrites[i][0] # extract front from tuple (front,distance) of index i
            den_branch_name = den_front.get_branch_name()
            b = den_branch_name.find("root")
            den_root_name = den_branch_name[:b+5]
            if den_root_name == self_root_name: # front on the same dendritic tree --> remove from the list
                del_candidates.append(samecell_dendrites[i])

        for del_item in del_candidates: # 2nd loop indexing del_candidates to remove item from the original list
            samecell_dendrites.remove(del_item)
        
        sorted_samecell_dendrites = sorted(samecell_dendrites, key=lambda e:abs(e[0].end.y-self.end.y))  # re-arrange order by distance in y-directions
        
        if (other_dendrites) or (samecell_dendrites):
            if (other_dendrites) and (samecell_dendrites): # if both detected choose closer one
                other = sorted_other_dendrites[0]
                samecell = sorted_samecell_dendrites[0]
                if other[1] <= samecell[1]: # comparing distance
                    other_dendrite = other # assign other as other_dendrite
                else:
                    other_dendrite = samecell # assign samacell as other_dendrite
        
            elif other_dendrites:
                # get first detected one only (not considering other dendritic structure still close to each other)
                other_dendrite = sorted_other_dendrites[0]
                if (verbose >= 4) or (verbose_cell == somaID):
                    print (ID,self.get_neuron_name(constellation),"extend_basal_dendrite, repulsion: list of other_dendrites_____self.orig =",self.orig,"self.end =",self.end)
                    for den in sorted_other_dendrites[0:min(3,len(sorted_other_dendrites))]:
                        den_front = den[0]
                        print ("_________________________",den_front.get_id(),den_front.get_neuron_name(constellation),"orig =",den_front.orig,"end =",den_front.end,", distance =",den[1])
            
            elif samecell_dendrites:
                other_dendrite = sorted_samecell_dendrites[0]
                if (verbose >= 4) or (verbose_cell == somaID):
                    print (ID,self.get_neuron_name(constellation),self.get_branch_name(),"extend_basal_dendrite, repulsion: list of samecell_dendrites_____self.orig =",self.orig,"self.end =",self.end)
                    for den in sorted_samecell_dendrites[0:min(3,len(sorted_samecell_dendrites))]:
                        den_front = den[0]
                        print ("_________________________",den_front.get_id(),den_front.get_neuron_name(constellation),den_front.get_branch_name(),"orig =",den_front.orig,"end =",den_front.end,", distance =",den[1])
            
            
            dir_to_repel = Point(self.end.x, other_dendrite[0].end.y, self.end.z) - self.end
            y_distance = abs(other_dendrite[0].end.y - self.end.y)

            # give repulson directly relative to distance
            repel_vec = dir_to_repel.norm()*(repel_fac/y_distance)
            if abs(repel_vec.y) >= repel_lim: # prevent from getting large repel value when other_dendrite is too close (e.g. y_distance < 0.1)
                repel_vec = dir_to_repel.norm()
            if (verbose >= 4) or (verbose_cell == somaID):
                    print (ID,"extend_basal_dendrite,added repel_vec, y_distance =",repel_vec,y_distance)
        
        else: # no neighboring dendrite found
            repel_vec = Point(0,0,0)
        
        # continue growth
        # search for PF around
        if self.order >= 1:
            #check if has synapse (come back from return on solve_collision_PC_den)
            if self.has_synapse():# self already has_synapse (skip target detection)
                synapse = self.get_synapse(constellation)
                front_id = synapse.pre_syn
                pre_syn_front = constellation.front_by_id(front_id)

            else:
                targets = self.get_fronts(constellation,what='name',name="granule",max_distance=5)
                if not targets:
                    heading_dir = self.end - self.orig
                
                    extension = self.unit_heading_sample()
                    
                    if heading_dir.z < 0: # growing downward
                        upwards = Point(0,0,0.6)
                    else:
                        upwards = Point(0,0,0.2)
                    new_pos = self.end + (extension + upwards) * (PC_growth_step/2) - repel_vec
                    
                    if (verbose >= 4) or (verbose_cell == somaID):
                        print (ID," : _extend_basal_dendrite_, no target is detected, grow to heading + upwards, return confirm_extension,heading_dir/upwards =",heading_dir,upwards,"self.end & new_pos =",self.end,new_pos)
                    self.confirm_extension(constellation,new_pos,ID,somaID)
                    return
                else: # detected targets
                    sorted_target = () # tuple
                    alt_target = None # store descending axon as an alternative target
                    for t in targets:
                        candidate = t[0] # extract only front from list of tuples (front,distance)
                        soma = self.get_soma(constellation)
                        if candidate.orig.z < soma.orig.z:
                            continue
                        if (candidate.swc_type == 1) or (candidate.swc_type == 12): # soma or filipod
                            continue
                        if candidate.has_synapse(): # candidate already has a synapse
                            continue
                        if candidate.swc_type == 2: # descending axon, only thin dendrites can make synapse with descending axon (Gundappa-Sulur, De Schutter et al. 1999)
                            continue
                        if candidate.orig.z < self.end.z:  # GC front is at lower than self
                            continue
                        if ((candidate.swc_type == 100) or (candidate.swc_type == 101)) and (abs(candidate.orig.x - candidate.end.x) >= 1.0): # skip weird behaving PF
                            if (verbose >= 3) or (verbose_cell == somaID):
                                print (ID,"candidate =",candidate.get_id(),"is oriented in weird way, abs =",abs(candidate.orig.x - candidate.end.x),"at", candidate.end)
                            continue
                        sorted_target += t # t = (front,distance)
                        break
            
                    if len(sorted_target) == 0:
                        heading_dir = self.end - self.orig
                        extension = self.unit_heading_sample()
                        if heading_dir.z < 0: # growing downward
                            upwards = Point(0,0,0.6)
                        else:
                            upwards = Point(0,0,0.2)
                        new_pos = self.end + (extension+upwards) * (PC_growth_step/2) - repel_vec
                        self.confirm_extension(constellation,new_pos,ID,somaID)
                        return
            
                    else: # has sorted_target
                        pre_syn_front = sorted_target[0]
                        distance = sorted_target[1]
                        if distance <= syn_distance: # close enough for new synapse
                            self.add_synapse(constellation,pre_syn_front,1.0,presynaptic=False) # self is postsynaptic front
                        else: # grow towards the target (the shortest path)
                            close_dist, close_point = dist3D_point_to_cyl(self.end,pre_syn_front.orig,pre_syn_front.end,points=True)
                            orig_cpoint_dir = pre_syn_front.orig - close_point
                            orig_cpoint_len = orig_cpoint_dir.length()
                            sur_pos = orig_cpoint_len / pre_syn_front.length()
                            goal = pre_syn_front.surface_point_to(self.end,mid=False,offset=PC_root_radius+f_radi,pos=sur_pos)
                            
                            direction = goal - self.end
                            n_direction = direction.norm()
                            distance = direction.length()
                            new_pos = self.end + n_direction * (distance - syn_distance*0.5)
                            y_dev = abs(self.end.y - new_pos.y)
                            y_dev2 = max(y_dev,1)
                            
                            if y_dev > repel_lim:
                                upwards = Point(0,0,y_dev2)
                                ndir_up = n_direction+upwards
                                new_pos = self.end + ndir_up.norm() * (distance - syn_distance*0.5)  - repel_vec

                            # check if new_pos deviate from heading direction
                            new_pos_dir = n_direction * (distance - syn_distance*0.5) # direction vector from [0,0,0]
                            ref_dir = Point(0,0,PC_growth_step)
                            heading = self.end - self.orig
                            heading_dir = heading.norm() * PC_growth_step
                            heading_angle = angle_two_dirs(ref_dir,heading_dir)
                            newpos_angle = angle_two_dirs(heading_dir,new_pos_dir)

                            if (self.prev_branch == 0) and (newpos_angle > heading_angle):
                                if (verbose >= 4) or (verbose_cell == somaID):
                                    print (ID," : _extend_basal_dendrite_, target_detected and grow closer to the target =",pre_syn_front.get_id(),", but deviated from headng_dir, return directed_branching, headinf_angle =",heading_angle,", newpos_angle =",newpos_angle,"self.end =",self.end,",goal =",goal,"repel_vec =",repel_vec,"new_pos =",new_pos)
                                self.directed_branching(constellation,new_pos,ID,somaID)
                                return

                            else: # prev_branch > 1 or newpos_angle is not deviating from heading --> grow closer to the target
                                self.confirm_extension(constellation,new_pos,ID,somaID)
                                return

            # added synapse with satisfactory target & grow perpendicular to that target
            # set parpendicular growth (Nagata,Ono,Kawana et al., 2006) using plane equation
            PF_vec = pre_syn_front.end - pre_syn_front.orig
            
            count = 0
            col_error = None
            
            x,y,z = symbols('x y z')
            eq = PF_vec.x * (x - self.end.x) + PF_vec.y * (y - self.end.y) + PF_vec.z * (z - self.end.z)
            # check direction of heading and growing to target
            heading_dir = self.end - self.orig
            close_dist, close_point = dist3D_point_to_cyl(self.end,pre_syn_front.orig,pre_syn_front.end,points=True)
            orig_cpoint_dir = pre_syn_front.orig - close_point
            orig_cpoint_len = orig_cpoint_dir.length()
            sur_pos = orig_cpoint_len / pre_syn_front.length()
            
            goal = pre_syn_front.surface_point_to(self.end,mid=False,offset=PC_root_radius+f_radi,pos=sur_pos)
            goal_dir = goal - self.end
            if math.copysign(1,heading_dir.x) == math.copysign(1,goal_dir.x):
                direction = goal_dir
            else: # goal is opposite to heading direction, still growing to the goal though having verbose to alert
                direction = goal_dir
                if (verbose >= 4) or (verbose_cell == somaID):
                    print(ID,"extend_basal_dendrite: perpendicular growth, goal is opposite to heading_dir, self.end =",self.end,",goal =",goal,"repel_vec =",repel_vec)

            n_direction = direction.norm() - repel_vec
            x_value = self.end.x + n_direction.x * PC_growth_step
            
            while count < 50:
                z_value = np.random.uniform(self.end.z, self.end.z + PC_growth_step) # random float: self.end.z <= z < self.end.z + growth_step
                
                eq2 = eq.subs(x, x_value).subs(z, z_value)
                
                given_ys = solve(eq2) # return list of y values
                
                if not given_ys: # PF_vec.y == 0 (rare case)
                    if (verbose >= 3) or (verbose_cell == somaID):
                        print (ID,"PC_extend_basal_dendrite, make synapse: no given_ys (rare case), assign self.end.y as given_y")
                    given_y = self.end.y
                else:
                    given_y = float(given_ys[0]) # convert to float
            
                new_pos = Point(x_value,given_y,z_value) - repel_vec
                if abs(new_pos.y - self.end.y) > repel_lim: # growing to y_direction more than upper limit of repel
                    if (verbose >= 3) or (verbose_cell == somaID):
                        print(ID,"extend_basal_dendrite: trying to grow on y_direction, self.end =",self.end,",new_pos =",new_pos)

                try:
                    new_front = self.add_child(constellation,new_pos,radius=PC_root_radius,swc_type=3)
                    if self.prev_branch > 0:
                        new_front.prev_branch = self.prev_branch - 1
                    self.disable(constellation)
                    return # success
                except CollisionError as error:
                    count += 1
                    col_error = error
                    continue
                except (GridCompetitionError, InsideParentError, VolumeError):
                    count += 1
                    continue

            
            if col_error:
                self.solve_collision_PC_den(constellation,new_pos,col_error,ID,somaID)
                return
            else:
                if (verbose >= 3) or (verbose_cell == somaID):
                    print (ID,"PC_extend_basal_dendrite, make synapse: could not find valid point on the plane, and not collision error --> disabled (few loop cases can happen, e.g. 6 cells/3000cells)")
                self.disable(constellation)
                return

        else: # not branched yet
            extension = self.unit_heading_sample()
            upwards = Point(0,0,0.3)
            new_pos = self.end + (extension + upwards) * PC_growth_step/2 - repel_vec
            if (verbose >= 4) or (verbose_cell == somaID):
                print (ID,"extend_basal_dendrite: not branched yet, self.end & new_pos =",self.end,new_pos)
            self.confirm_extension(constellation,new_pos,ID,somaID)
            return

        print ("EROOR! in extend_basal_dendrite: ending without return. ID =",ID,self.has_synapse(),self.order,col_error)

    def confirm_extension(self,constellation,new_pos,ID,somaID):
        
        new_pos_dir = new_pos - self.end # direction vector from [0,0,0]
        if abs(new_pos.y - self.end.y) > repel_lim: # growing to y_direction more than upper limit of repel
            if (verbose >= 3) or (verbose_cell == somaID):
                print(ID,"confirm_extension: trying to grow on y_direction, self.end =",self.end,",new_pos =",new_pos)
    
        ref_dir = Point(0,0,PC_growth_step)
        heading = self.end - self.orig
        heading_dir = heading.norm() * PC_growth_step
        heading_angle = angle_two_dirs(ref_dir,heading_dir)
        newpos_angle = angle_two_dirs(heading_dir,new_pos_dir)

        if (self.prev_branch == 0) and (newpos_angle > heading_angle): # growing on the plane, but deviated from headng_dir
            self.directed_branching(constellation,new_pos,ID,somaID)
            return

        try:
            new_front = self.add_child(constellation,new_pos,radius=PC_root_radius,swc_type=3)
            if self.prev_branch > 0:
                new_front.prev_branch = self.prev_branch - 1
            self.disable(constellation) # stop growing
            return # success
        except VolumeError: # no attempt to correct error
            if (verbose >= 3) or (verbose_cell == somaID):
                print (self.get_branch_name(),ID,"confirm_extension new_pos is out of simulation")
            self.disable(constellation) # stop growing
            return
        except InsideParentError: # no attempt to correct error
            if (verbose >= 3) or (verbose_cell == somaID):
                print (self.get_branch_name(),ID,"confirm_extension new_pos is inside parent")
            return
        except CollisionError as error: # attempt to find solution
            self.solve_collision_PC_den(constellation,new_pos,error,ID,somaID)
            return

        except GridCompetitionError:
            if (verbose >= 3) or (verbose_cell == somaID):
                print (self.get_branch_name(),ID,"confirm_extension new_pos is raising a GridCompetitionError")
            return


    def branching(self,constellation,ID,somaID):
        col_front = None
        
        # sample first branch on heading direction
        count = 0
        new_name = self.get_branch_name() + "_0"
        
        if abs(self.end.y - self.orig.y) > repel_lim: # stronger upward force applys if grwoing to y
            upwards = Point(0,0,PC_growth_step)
        else:
            upwards = Point(0,0,0.05)
    
        while count < 100:
            extension = self.unit_heading_sample(width=1.0,max_angle=20)
            headup = extension + upwards
            first_pos = self.end + headup.norm() * (PC_growth_step)
            success = False
            try:
                new_front1 = self.add_child(constellation,first_pos,radius=PC_root_radius,\
                                                            branch_name=new_name,swc_type=3)
                success = True
                break
            except CollisionError as error:
                col_front = error.collider
                count += 1
                continue
            except (GridCompetitionError, InsideParentError, VolumeError):
                count += 1
                continue
    
        if not success:
            if (verbose >= 4) or (verbose_cell == somaID):
                print (ID," : _branching_, could not find first point for branching return, self.end =",self.end)
                if col_front:
                    print ("---> colliding with id =", col_front.get_id(),", swc =",col_front.swc_type)
            return
        
        # calculate degree deviation of 1st pos from z-plane
        opp_direction = first_pos - Point(first_pos.x,first_pos.y,self.end.z)
        opp = opp_direction.length()
        hyp_direction = first_pos - self.end
        hyp = hyp_direction.length()
        dev_theta = np.arcsin(opp/hyp)
    
        # sampling 2nd branch
        count2 = 0
        new_name = self.get_branch_name() + "_1"
        while count2 < 100:
            random_degree = np.random.normal(degree,20) # arbitrary
            theta = random_degree * np.pi / 180
            theta_options = [theta,-theta] # try both positive & negative directions of theta
            np.random.shuffle(theta_options) # avoid keep having the same branch directions
            for theta in theta_options:
                new_pos = self.end +  Point(np.sin(dev_theta+theta),0,\
                                            np.cos(dev_theta+theta))*(PC_growth_step)
                success = False
                try:
                    new_front2 = self.add_child(constellation,new_pos,radius=PC_root_radius,\
                                                                branch_name=new_name,swc_type=3)
                    success = True
                    break
                except CollisionError as error:
                    col_front = error.collider
                    continue
                except (GridCompetitionError, InsideParentError, VolumeError):
                    continue

            if success:
                break
            else:
                count2 += 1
        
        if not success:
            if (verbose >= 4) or (verbose_cell == somaID):
                print (somaID,ID," : _branching_, could not find 2nd point, extend 1st point only, self.end =",self.end, "front1.end =",new_front1.end)
            self.disable(constellation)
            return
        else:
            if (verbose >= 4) or (verbose_cell == somaID):
                ydev1 = abs(self.end.y - new_front1.end.y)
                ydev2 = abs(self.end.y - new_front2.end.y)
                if (ydev1 >= repel_lim) or (ydev2 >= repel_lim):
                    print (ID,"PC_branching success, self, front1, front2 =", self.end, new_front1.end, new_front2.end)
        
            # prevent newly branching front to branch again soon
            new_front1.prev_branch = 2
            new_front2.prev_branch = 2
            
            self.disable(constellation)
            return
    

    def directed_branching(self,constellation,new_pos,ID,somaID):
        
        # growing to y_direction more than upper limit of repel, give stronger upward force
        if abs(self.end.y - self.orig.y) > repel_lim:
            upwards = Point(0,0,PC_growth_step)
        else:
            upwards = Point(0,0,0.05)
        
        count = 0
        new_front1 = False
        new_front2 = False
        new_name = self.get_branch_name() + "_0"
        while count < 100: # to deal with GridCompetitionError
            heading_dir = self.unit_heading_sample(width=1.0,max_angle=20)
            headup = heading_dir + upwards
            heading_pos = self.end + headup.norm() * (PC_growth_step)
            try:
                new_front1 = self.add_child(constellation,heading_pos,radius=PC_root_radius,\
                                                            branch_name=new_name,swc_type=3)
                success = True
                break
            except CollisionError as error:
                col_front = error.collider
                break # not remedied
            except (InsideParentError, VolumeError):
                break # not remedied
            except GridCompetitionError:
                count += 1
                continue
        count = 0
        new_front2 = False
        new_name = self.get_branch_name() + "_1"
        while count < 100: # to deal with GridCompetitionError
            try:
                new_front2 = self.add_child(constellation,new_pos,radius=PC_root_radius,\
                                                            branch_name=new_name,swc_type=3)
                success = True
                break
            except CollisionError as error:
                col_front = error.collider
                break # not remedied
            except (InsideParentError, VolumeError):
                break # not remedied
            except GridCompetitionError:
                count += 1
                continue
    
        if new_front1 and new_front2: # 2 positions may be valid
            if (verbose >= 3) or (verbose_cell == somaID):
                ydev1 = abs(self.end.y - new_front1.end.y)
                ydev2 = abs(self.end.y - new_front2.end.y)
                if (ydev1 >= repel_lim) or (ydev2 >= repel_lim):
                    print (ID,"PC_directed_branching return b, self, front1, front2 =", self.end, new_front1.end, new_front2.end)
            # prevent newly branching front to branch again soon
            new_front1.prev_branch = 2
            new_front2.prev_branch = 2
            
            self.disable(constellation)
            return

        elif not new_front1 and not new_front2: # both branch failed
            new_pos = self.end + heading_dir.norm() * PC_growth_step
            self.prev_branch = 2 # avoid loop between confirm_extension & directed_branching
            if (verbose >= 4) or (verbose_cell == somaID):
                print (ID," : _directed_branchinf_, did not get 2 valid points & extend to heading, self.end =",self.end,", new_pos =",new_pos," --> return confirm_extension")
            self.confirm_extension(constellation,new_pos,ID,somaID)
            return


        else: #either position is invalid, only extend the success one
            if (verbose >= 4) or (verbose_cell == somaID):
                print (somaID,ID," : _directed_branching_, could not find either point extend single front only, self.end =",self.end)
            self.disable(constellation)
            return
        
    def solve_collision_PC_den(self,constellation,new_pos,error,ID,somaID):
    
        if error:
            col_front = error.collider
        else:
            if (verbose >= 4) or (verbose_cell == somaID):
                print (somaID,ID," : _solve_collision_PC_den_ no col_fronts were detected, disable & return")
            self.disable(constellation)
            return
        
        points = self.solve_collision(constellation,new_pos,error)
        
        if points:
            soma = self.get_soma(constellation)
            if points[-1].z >= (soma.orig.z): # <condition 1> check if the last point is higher than the bottom of PC soma
                heading_dir = self.end - self.orig
                new_dir = points[0] - self.end
                if math.copysign(1,heading_dir.x) == math.copysign(1,new_dir.x): # <condition 2> check if the path to reach new_pos is not deviating from heading_dir
                    try:
                        new_fronts = self.add_branch(constellation,points,swc_type=3)
                        
                        if self.prev_branch > 0:
                            for front in new_fronts:
                                front.prev_branch = max(0,self.prev_branch - 1)
                        self.disable(constellation) # success -> disable this filipod front
                        return
                    except (CollisionError,GridCompetitionError,InsideParentError,\
                                                        VolumeError):
                        if (verbose >= 3) or (verbose_cell == somaID): # appears around 4 in one simulation
                                print (somaID,ID," : _solve_collision, has points but they did not satisfy given conditions", col_front.get_neuron_name(constellation),col_front.swc_type,"(ID =", col_front.get_id(),"), self.end =",self.end,", paths = ",points[0],"new point =", points[-1])
                        self.disable(constellation)
                        return
        
        # check whether this is first or second failure
        if self.is_status1(): # second failure
            if (verbose >= 4) or (verbose_cell == somaID):
                print (ID,": _solve_collision_PC_den, could not find points by solve collision, colliding partner =", col_front.get_neuron_name(constellation),col_front.swc_type," --> disable")
            self.disable(constellation)
            return
        else:
            self.set_status1() # flag as first failure
            return


class BergmannGlia(Front):
    _fields_ = Front._fields_ + [('occupancy', c_int)]
    
    def manage_front(self,constellation):
        if self.swc_type == 1:
            self.Bergmann_soma(constellation)
        else:
            self.Bergmann_glia(constellation)
    
    # make initial glia roots from soma
    def Bergmann_soma(self,constellation):
        n_roots = 0
        for i in range(0, num_roots):
            name = self.get_neuron_name(constellation)+'proc'+str(i)
            count = 0
            success = False
            while count < 20:
                # increase noise gradually
                new_pos = self.orig + \
                            Point(BG_offsets[i],0.,root_len) + random_point(-count/20,count/20)
                try:
                    new_front = self.add_child(constellation,new_pos,radius=BGproc_radi,\
                                                            branch_name=name,swc_type=99)
                    success = True
                    n_roots += 1
                    break # success

                except (GridCompetitionError, InsideParentError, VolumeError,CollisionError) as error:
                    count += 1
                    continue # pick another point, no attempt to correct the error

            if not success:
                if (verbose >= 5) or (verbose_cell == self.get_id()):
                    print (self.get_id(),": could not find solution points for swc99 skip ",name)
    
        if (verbose >= 5) or (verbose_cell == self.get_id()):
            print (name," (",self.get_id(),") total process ____",n_roots)
        self.disable(constellation)
        
    # extend glia roots
    def Bergmann_glia(self,constellation):
        
        if self.end.z > pia: # BG process reached to the end
            if (verbose >= 5) or (verbose_cell == self.get_soma(constellation,returnID=True)):
                print (self.get_soma(constellation,returnID=True),self.get_id(),"location of Bergman glia tip", self.end)
            self.disable(constellation) # stop growing

        # swc == 99 or 7
        else:
            dir_to_pia = Point(0.,0.,1.)
            new_pos = self.end + dir_to_pia * proc_len
            count = 0
            while count < 10:
                try:
                    new_front = self.add_child(constellation,new_pos,swc_type=7)
                    self.disable(constellation) # stop growing
                    return # success
                except (VolumeError, InsideParentError): # no attempt to correct error
                    self.disable(constellation) # stop growing
                    return
                except CollisionError as error: # attempt to find solution
                    count += 1
                    continue

                except GridCompetitionError:
                    count += 1
                    continue # try again
            # did not return -> extension failure
            if constellation.cycle - self.birth > 1 : # second cycle we tried
                if (verbose >= 5) or (verbose_cell == self.get_soma(constellation,returnID=True)):
                    print (self.get_branch_name(),self.get_id()," (",self.get_soma(constellation,returnID=True),") : could not solve Bergman collision, stop at z =",self.end.z)
                self.disable(constellation) # stop growing


class GranuleCell(SynFront):
    _fields_ = Front._fields_ + [('mode',c_int),('target_ID',ID),\
                ('loop_lim',c_int),('root_pos',Point),\
                ('nograndchildD_loop',c_int)]
    
    def manage_front(self,constellation):
        
        # adjust close_enough by initiation timing
        name = self.get_neuron_name(constellation)
        if (name.startswith('granule_p12')) or (name.startswith('granule_p11')) \
            or (name.startswith('granule_p10')):
            close_enough = 0.2 + 0.45
        elif (name.startswith('granule_p9')) or (name.startswith('granule_p8')) \
            or (name.startswith('granule_p7')):
            close_enough = 0.2 + 0.30
        elif (name.startswith('granule_p6')) or (name.startswith('granule_p5')) \
            or (name.startswith('granule_p4')):
            close_enough = 0.2 + 0.15
        else:
            close_enough = 0.2

        ID = self.get_id()

        if self.swc_type == 1:  # soma
            self.gc_soma(constellation,close_enough,axon_root_len,ID)
            return
        else: # all other types of front
            soma = self.get_soma(constellation)
            somaID = soma.get_id()
            # pass the conditions stored in soma to front
            self.mode = soma.mode
            self.target_ID = soma.target_ID
            if self.mode == -1:
                self.disable(constellation)
                return

        if self.swc_type == 12:   # filipod
            if self.mode == 1:
                self.extend_filipod(constellation,close_enough,ID,somaID)
                return
            elif (self.mode == 2) or (self.mode == 3):
                target_front = constellation.front_by_id(self.target_ID)
                goal = target_front.surface_point_to(self.end)
                direction_xy = Point(goal.x,goal.y,self.end.z) - self.end
                distance = direction_xy.length()
                if distance <= close_enough:
                    self.radial_filipod(constellation,close_enough,ID,somaID)
                    return
                else:
                    self.extend_filipod(constellation,close_enough,ID,somaID)
                    return
            elif self.mode == 4:
                self.after_guidance(constellation,ID,somaID,close_enough)
                return
            elif (self.mode == 5) or (self.mode == -1):
                self.disable(constellation)
                return
            else:
                print ("ERROR! in manage_front: swc = 12, unexpected mode = ", self.mode)
        elif self.swc_type == 2 and self.is_status1():  # vertical axon
            self.axon_branch(constellation,ID,somaID,close_enough)
            return
        elif (self.swc_type == 100) or (self.swc_type == 101): # parallel fibers
            self.simple_PF_extension(constellation,ID,somaID,close_enough)
            return
        else:
            print ("ERROR! in manage_front, unexpected swc_type, somaID =",somaID,"swc_type = ",self.swc_type,",selfID =",ID,",status_1 =",self.is_status1(),"--> disabled")
            self.disable(constellation)
            return


    def gc_soma(self,constellation,close_enough,axon_root_len,ID):

        # Get child info
        children = self.get_children(constellation)
        filipod = None
        axon = None
        for child in children:
            if child.swc_type == 12:
                if filipod:
                    print ("ERROR! in gc_soma: more than 2 filipod children",ID,len(children),"in mode =", self.mode)
                filipod = child
            elif child.swc_type == 2:
                if axon:
                    print ("ERROR! in gc_soma: more than 2 axon children",ID,len(children))
                axon = child
            else:
                print ("ERROR! in gc_soma: unexpected swc_type = ",child.swc_type,ID)
    
        # failed GC, retract self after retracting all children for whom could not even reach PCL
        if (self.mode == -1) and (self.orig.z > PCL_z):
            if filipod:
                self.retract_branch(constellation,filipod)
                return
            elif axon:
                self.retract_branch(constellation,axon)
                return
            else: # only soma left
                self.retract(constellation)
                return
        
        # make root axon if appropriate
        if (self.mode == 2) and (not axon) and (self.orig.z > PCL_z):
                target_front = constellation.front_by_id(self.target_ID)
                target_point = Point(target_front.end.x, target_front.end.y,\
                                     self.orig.z)
                goal = target_front.surface_point_to(self.orig)
                soma_direction_xy = Point(goal.x,goal.y,self.orig.z) - self.orig
                soma_distance = soma_direction_xy.length()
                # make the initial axon
                if soma_distance <= close_enough:
                    # decide the x direction away from target BG proc
                    if target_point.x - self.orig.x < 0:
                        root_x = self.orig.x + axon_root_x_coord
                    else:
                        root_x = self.orig.x - axon_root_x_coord
                    direction = Point(root_x, self.orig.y, \
                                            self.orig.z + 2.0) - self.orig
                    new_pos = self.orig + direction.norm()*axon_root_len
                    if self.rootaxon_initiation(constellation,target_point,\
                                            new_pos,direction,False,ID,ID):
                        self.mode = 3
                        # set initial firing rate
                        neuron = self.get_neuron(constellation)
                        rate = np.random.random() * MAX_FR
                        neuron.set_firing_rate(constellation,rate)
                        return
                    else: # try once more with noise added
                        if self.rootaxon_initiation(constellation,\
                                    target_point,new_pos,direction,True,ID,ID):
                            self.mode = 3
                            # set initial firing rate
                            neuron = self.get_neuron(constellation)
                            rate = np.random.random() * MAX_FR
                            neuron.set_firing_rate(constellation,rate)
                            return
        

        # Migrate if possible
        if filipod: # if mode < 4 a filipod grandchild is needed
            if (self.mode == 5) or (filipod.num_children == 1) or (self.mode == -1): # migrate
                count = 0 # loop for GridCompetitionError
                while count < 50:
                    if axon: # filipod + trailing axon mode
                        try:
                            coordinate = self.migrate_soma(constellation,None,\
                                            filipod=True,trailing_axon=True)
                            return # filipod + axon success -> self.has_moved() = True
                        except (GridCompetitionError):
                            count += 1
                            continue # try again
                        except (CollisionError,ActiveChildError):
                            # filipod + axon failed
                            if not self.has_moved(): # second failed attempt: delete filipod
                                self.retract_branch(constellation,filipod)
                                return
                            else:
                                count += 1
                                continue

                    else: # only filipod
                        try:
                            coordinate = self.migrate_soma(constellation,None,\
                                                            filipod=True)
                            # filipod success -> self.has_moved() = True
                            return
                        except (GridCompetitionError):
                            count += 1
                            continue # try again
                        except (CollisionError,ActiveChildError):
                            # filipod failed
                            if not self.has_moved(): # second failed attempt: delete filipod
                                self.retract_branch(constellation,filipod)
                                return
                            else:
                                count += 1
                                continue
                
                # end of while loop
                if not self.has_moved():# second failed attempt: delete filipod
                    self.retract_branch(constellation,filipod)
        
                return

            elif (self.mode < 5) and (filipod.num_children == 0): # no grand_child
                if (self.mode == 2) and (self.orig.z <= IGL_z): # reached IGL_z without extending axon
                    if (verbose >= 6) or (verbose_cell == ID):
                        print (ID," : gc_soma, reached IGL_z without axon, have 1 filipod --> mode -1")
                    self.mode = -1
                    return
                
                self.nograndchildD_loop += 1 # avoid endless no migration
                
                if self.nograndchildD_loop > 6: # no grandchild for a while and stuck
                    if ((verbose >= 6) and (self.orig.z > PCL_z)) or (verbose_cell == ID):
                        print (ID," : gc_soma, nograndchildD_loop reached its limit in mode = ", self.mode, self.nograndchildD_loop,"stop migration at",self.orig,self.get_neuron_name(constellation))
                        print ("    its filipod",filipod)
                    self.mode = -1 # migrate to a filipos then stop migrating next cycle
                    return
                return

            else:
                print ("ERROR! undefined condition in gc_soma with filipod, ID =",ID,", filipod.num_children =",filipod.num_children,", mode =",self.mode)

        else: # no filipod -> try to make one
            if (self.mode == 5) or (self.mode == -1): # reached IGL or stuck below PCL
                self.disable(constellation) # stop migrating
                return

            elif self.mode == 0:
                self.radial_target(constellation,close_enough,ID)
                return

            else: # no filipod in mode 1,1.5,2,3 --> extend new filipod
                if self.mode == 1:
                    self.extend_filipod(constellation,close_enough,ID,ID)
                    return
                elif (self.mode == 3) or (self.mode == 2):
                    self.radial_filipod(constellation,close_enough,ID,ID)
                    return
                elif self.mode == 4:
                    self.after_guidance(constellation,ID,ID,close_enough)
                    return
                else:
                    print ("ERROR! in gc_soma: unexpected mode = ",self.mode,"(",ID,")")


        print("ERROR in gc_soma: reached bottom without returns", ID)


    
    def radial_target(self,constellation,close_enough,somaID):

        if (self.swc_type == 1) and (self.mode == 0):
            guidance = self.get_fronts(constellation,what='name',\
                name="bergmann",max_distance=15,swc_types=[7]) #GC soma in the bottom of EGL can travel 90µm (Komuro et al., 2001)
                # max_distance=20: 40 to 110 guidance found
                # max_distance=15: 10 to 120 guidance found
                # max_distance=10: 0 to 4 guidance found
            num_list = 1 # just get one target BG process
            guidance_len = len(guidance)
            guidance_list = []
            name_list = []
            #print (somaID," : _radial_target_, guidance_len =",guidance_len)
        
        else:
            print ("ERROR! in radial_target: unexpected swc and mode, swc =", self.swc_type," mode =",self.mode,"(",somaID,")")

        num = 0
        i = 0
        while (num < num_list) and (i < guidance_len):
            BG = guidance[i][0]
            process = BG.get_branch_name()
            BGid = BG.get_id()
            if process in name_list:
                i +=1
                continue
            name_list.append(process)
            BGroot = BG
            # memorize occupancy at BG root front
            while BGroot.swc_type != 99:
                BGroot = BGroot.get_parent(constellation)
            i +=1
            if BGroot.occupancy > 15:
                if (verbose >= 5) or (verbose_cell == somaID):
                    print (somaID," : _radial_target_ while loop detected",BGroot.get_branch_name(),"is occupied and skipped ")
                i +=1
                continue
            if (BG.orig.z <= self.orig.z):
                guidance_list.append(BGid)
                num +=1

        #for safety, if not find enough
        if num < num_list:
            num_list = num

        #for safety, if no process found
        if num == 0:
            print ("ERROR! in radial_target",somaID,"could not find open BG process ")
            self.mode = -1 # self is soma
            self.disable(constellation)
            return

        # got target
        self.target_ID = guidance_list[0] # self is soma

        self.mode = 1 # self is soma

        BGroot.occupancy += 1 # may not be accurate due to competition
        # below arise LockError
        """
        if constellation.lock(BGroot):
            BGroot.occupancy += 1
            result = constellation.unlock(BGroot)
        """
        
        self.extend_filipod(constellation,close_enough,ID,somaID)
        return


    def extend_filipod(self,constellation,close_enough,ID,somaID):
        target_front = constellation.front_by_id(self.target_ID)
        if self.swc_type == 1:
            z = self.orig.z
            self_pos = self.orig
        else:
            z = self.end.z
            self_pos = self.end
        goal = target_front.surface_point_to(self_pos)
        direction_xy = Point(goal.x,goal.y,self_pos.z) - self_pos
        distance = direction_xy.length()

        if distance <= close_enough: # close enough
            if (self.mode == 0) or (self.mode == 1):
                self.mode = 2
                if self.swc_type == 12:
                    soma = self.get_soma(constellation)
                    soma.mode = self.mode # avoiding LockError
                    """
                    if constellation.lock(soma): # lock the soma before changing its attribute
                        soma.mode = self.mode
                        result = constellation.unlock(soma) # unlock the soma again
                    """

            self.radial_filipod(constellation,close_enough,ID,somaID)
            return
        
        # mode 1.5 or 2 can come to this condition again
        elif distance <= close_but_closer: # close but grow closer
            dir_to_guidance = direction_xy.norm() * (distance*0.2)

        # mode 1.5 or 2 also come to this
        else:# far, grow closer with stronger weights
            dir_to_guidance = direction_xy.norm() * (distance*0.4)# less than 0.3 then new_pos is inside sphere when distance 0.3 < d <= 0.33

        new_pos = self_pos + dir_to_guidance
        # check if the filipod is longer than radius of soma
        if (self.swc_type == 1):
            filipod_direction = new_pos - self_pos
            filipod_len = filipod_direction.length()
            if filipod_len <= 0.1: # filipod is too short --> adjust length
                shortage = 0.1 - filipod_len
                new_pos += direction_xy.norm()*(shortage+0.001)

        try:
            new_front = self.add_child(constellation,new_pos,radius=f_radi,\
                                    swc_type=12,branch_name="filipod")
            if self.swc_type == 12: # filipod should stop growth
                self.disable(constellation)
            elif self.mode < 1:
                self.mode = 1 # self is soma
            #else:
                # swc=1 & mode=1
            return
        except CollisionError as error: # attempt to correct error
            if (self.swc_type == 1):
                self.check_validity_soma(constellation,new_pos,close_enough,\
                                            error,ID,somaID)
                return
            elif (self.swc_type == 12):
                self.check_validity_filipod(constellation,new_pos,\
                                                close_enough,error,ID,somaID,False)
                return
            else:
                print ("ERROR in extend_filipod unexpected swc and mode, swc =", self.swc_type, " in mode =",self.mode,", somaID =",somaID)
        except (GridCompetitionError, InsideParentError, VolumeError):
            return # no attempt to correct error


    def radial_filipod(self,constellation,close_enough,ID,somaID):
        if self.swc_type == 1:
            self_pos = self.orig
        else: # swc == 12
            self_pos = self.end
        
        if self_pos.z <  PCL_z: # dont have to be close to BG proc anymore (can ignore appropriate point sorting in CV_filipod)

            self.mode = 4
            if self.swc_type == 12:
                soma = self.get_soma(constellation)
                soma.mode = self.mode # avoiding LockError
                """
                if constellation.lock(soma): # lock the soma before changing its attribute
                    soma.mode = self.mode
                    result = constellation.unlock(soma) # unlock the soma again
                """
        
            self.after_guidance(constellation,ID,somaID,close_enough)
            return

        else:
            new_pos = self_pos + radial_step
            try:
                new_front = self.add_child(constellation,new_pos,radius=f_radi,\
                                        swc_type=12,branch_name="filipod")
                if self.swc_type == 12: # extension success & filipod should stop growth (have active new front)
                    self.disable(constellation)
                return
            except CollisionError as error: # attempt to correct error
                if (self.swc_type == 1):
                    self.check_validity_soma(constellation,new_pos,\
                                            close_enough,error,ID,somaID)
                    return
                elif (self.swc_type == 12):
                    self.check_validity_filipod(constellation,new_pos,\
                                                close_enough,error,ID,somaID,False)
                    return
                else:
                    print ("ERROR in radial_filipod unexpected swc and mode, swc =", self.swc_type, " in mode =",self.mode,", somaID =",somaID)
            except (GridCompetitionError, InsideParentError, VolumeError):
                return # no attempt to correct error


    def after_guidance(self,constellation,ID,somaID,close_enough):
        if self.end.z >= IGL_z:
            new_pos = self.end + ag_step
            try:
                new_front = self.add_child(constellation,new_pos,radius=f_radi,\
                                        swc_type=12,branch_name="filipod")
                if self.swc_type == 12: # success & filipod should stop growth
                    self.disable(constellation)
                return
            except CollisionError as error: # attempt to correct error
                if (self.swc_type == 1):
                    self.check_validity_soma(constellation,new_pos,\
                                            close_enough,error,ID,somaID)
                    return
                elif (self.swc_type == 12):
                    self.check_validity_filipod(constellation,new_pos,\
                                                close_enough,error,ID,somaID,False)
                    return
                else:
                    print ("ERROR in after_guidance unexpected swc and mode, swc =", self.swc_type, " in mode =",self.mode,", somaID =",somaID)
            except (GridCompetitionError, InsideParentError, VolumeError):
                # no attempt to correct error
                return
    
        else: # filipod arrived at IGL and Terminate
            self.mode = 5
            if self.swc_type == 12:
                soma = self.get_soma(constellation)
                soma.mode = self.mode # avoiding LockError
                """
                if constellation.lock(soma): # lock the soma before changing its attribute
                    soma.mode = self.mode
                    result = constellation.unlock(soma) # unlock the soma again
                """
        
            # set new firing rate
            neuron = self.get_neuron(constellation)
            rate = np.random.random() * NEW_FR
            neuron.set_firing_rate(constellation,rate)
            
            self.disable(constellation)
            return


    def axon_branch(self,constellation,ID,somaID,close_enough):
        children = self.get_children(constellation)
        for npf in range(2):
            if npf == 0:
                for child in children:
                    if child.swc_type == 100: # already made this front
                        continue
                new_pos = self.end - initial_pf_step
                swc_type = 100
            else:
                for child in children:
                    if child.swc_type == 101: # already made this front
                        self.disable(constellation) # axon should stop growth
                        return
                new_pos = self.end + initial_pf_step
                swc_type = 101
            
            name = 'pf_root' + str(swc_type) + \
                                self.get_neuron_name(constellation)
            
            count = 0
            col_error = None
            while count < 10:
                try:
                    new_front = self.add_child(constellation,new_pos,radius=f_radi,\
                                            swc_type=swc_type,branch_name=name)
                    break
                except (VolumeError, InsideParentError):
                    break
                except GridCompetitionError:
                    count += 1
                    continue # while loop
                except CollisionError as error: # attempt to correct error
                    col_error = error
                    break
            
            if not col_error: # made one branch or unsolvable error
                continue # for npf loop

            # CollisionError occurred
            CK = False
            Skip = False
            col_front = col_error.collider
            col_front_name = col_front.get_neuron_name(constellation)
            z_offset = 0.1

            if col_front_name.startswith('bergmann'):
                up = 0
                if col_front.swc_type == 1: #rare case
                    up = 3. + 0.5 # BGsoma radi + extra
                center = Point(col_front.orig.x, col_front.orig.y, \
                                            self.end.z +  z_offset + up)
                circle_radius = BG_GCoffset
                center2 = center
                circle_radius2 = circle_radius + close_enough + 0.2
            #horizontally oriented fronts
            elif (col_front.swc_type == 1 and self.mode <= 1) or (col_front.swc_type == 12 and self.mode <= 1) or (col_front.swc_type == 100) or (col_front.swc_type == 101):
                center = col_front.orig#
                circle_radius = col_front.radius + self.radius + 0.05
                center2 = center
                circle_radius2 = circle_radius + 0.25
            #vertically oriented fronts + PC dendrites
            elif (col_front.swc_type == 1 and self.mode > 1) or (col_front.swc_type == 12 and self.mode > 1) or (col_front.swc_type == 2) or (col_front.swc_type == 5) or (col_front.swc_type == 3):
                center = Point(col_front.orig.x, col_front.orig.y, self.end.z + z_offset)
                circle_radius = (col_front.radius + self.radius) * 1.1
                center2 = center
                circle_radius2 = circle_radius + 0.25
            else:
                print ("ERROR! in axon_branch : unexpected swc of col_front =",col_front.swc_type,"(",col_front_name,")")

            n = (2*np.pi*circle_radius)/0.2
            num_points = int(round(n))
            n2 = (2*np.pi*circle_radius2)/0.2
            num_points2 = int(round(n2))

            target_front = constellation.front_by_id(self.target_ID)
            target_point = Point(target_front.end.x,\
                                    target_front.end.y,self.orig.z)
            x_dis = target_point.x - self.orig.x
            soma = self.get_soma(constellation)

            for further in range(2):
                if further == 0:
                    n = (2*np.pi*circle_radius)/0.2
                    num_points = int(round(n))
                else:
                    center = center2
                    circle_radius = circle_radius2
                    n2 = (2*np.pi*circle_radius)/0.2
                    num_points = int(round(n2))

                point_list = [] # to use append
                num = 0
                num_point_memory = num_points
                point_counts = 0
                while num < num_point_memory:
                    nums = num_points*1.5
                    num_samples = int(round(nums))
                    points = col_front.alternate_locations(center,circle_radius,num_samples)
                
                    for p in points:
                        if p == self.end:
                            continue
                        if (swc_type == 100) and (p.y >= self.end.y):
                            continue
                        if (swc_type == 100) and (col_front.swc_type == 7) and (p.y >= col_front.orig.y):
                            continue
                        if (swc_type == 101) and (p.y <= self.end.y):
                            continue
                        # checking if the pos is far away enough from BG proc
                        if (x_dis < 0) and (p.x < self.end.x):
                            continue
                        if (x_dis >= 0) and (p.x > self.end.x):
                            continue
                        if p in point_list:
                            continue
                        if num >= num_point_memory:
                            break
                        point_list.append(p)
                        num +=1
                    point_counts +=1
                    if point_counts > 3: # less points given
                        num_point_memory = num
                #for safety
                if num == 0:
                    if (verbose >= 6) or (verbose_cell == somaID):
                            print (somaID,ID," : _axon_branch_ swc =",swc_type," no good point found, interfered by swc/ID =",col_front.swc_type,"/",col_front.get_soma(constellation,returnID=True))
                    if further == 0:
                        continue # go on to the next futher = 1 for-loop
                    else:
                        break
                else:
                    for new_pos in point_list:
                        try:
                            new_front = self.add_child(constellation,new_pos,\
                                    radius=f_radi,swc_type=swc_type,branch_name=name)
                            CK = True
                            break
                        except (CollisionError, GridCompetitionError, InsideParentError, VolumeError): # no attempt to correct error
                            continue

                if CK == True:
                    break # for-further-loop

                #couldnt find alternative points
                if (verbose >= 6) or (verbose_cell == somaID):
                    print (somaID,ID," : _axon_branch_ swc =",swc_type," is skipped, interfered by swc/ID =",col_front.swc_type,"/",col_front.get_soma(constellation,returnID=True))
                continue # for-further-loop
    
        if self.swc_type == 2: # axon should stop growth
            self.disable(constellation)
            return

        else:
            print ("ERROR! in axon_branch, unexpected swc_type =",self.swc_type,ID)
            return


    def simple_PF_extension(self,constellation,ID,somaID,close_enough):
        if self.swc_type == 100:
            new_pos = self.end - pf_step
        if self.swc_type == 101:
            new_pos = self.end + pf_step

        try:
            new_front = self.add_child(constellation,new_pos)
            self.disable(constellation)
            return
        except CollisionError as error: # attempt to correct error
            self.check_validity_PF(constellation,new_pos,close_enough,\
                                    error,ID,somaID)
            return
        except (VolumeError):
            self.disable(constellation) # stop growing
            return
        except (GridCompetitionError, InsideParentError):
            return # no attempt to correct error


    def check_validity_filipod(self,constellation,new_pos,close_enough,\
                                error,ID,somaID,short):
        col_front = error.collider
        target_front = constellation.front_by_id(self.target_ID)
        
        if short: # 2nd call, assign shorter new_pos
            if self.mode == 4:
                shorter_new_pos = self.end + (ag_step/3)
                new_pos = shorter_new_pos
            else:
                shorter_new_pos = self.end + (radial_step/3)
                new_pos = shorter_new_pos

            try:
                new_front = self.add_child(constellation,new_pos,\
                            radius=f_radi,swc_type=12,branch_name="filipod")
                # filipod should stop growth
                self.disable(constellation)
                return # success
            except CollisionError: # no attempt to correct error
                pass # go to the next code
            except (GridCompetitionError, InsideParentError, VolumeError):
                pass # no attempt to correct error

        points = self.solve_collision(constellation,new_pos,error)

        if points: # a list of points
            if (self.mode == 2) or (self.mode == 3) or (self.mode == 4):
                failed = False
                if points[-1].z > self.end.z:
                    failed = True
            
                if failed:
                    # repeated failures (have points but not appropriate)
                    if (constellation.cycle - self.birth) > 2:
                        if (verbose >= 6) or (verbose_cell == somaID):
                            print (somaID,ID,": _check_validity_filipod in mode = ",self.mode,", have points but not appropriate, disable at", self.end," interfered by swc/name=",col_front.swc_type,"/",col_front.get_soma(constellation,returnID=True),"located at", col_front.orig)
                        self.disable(constellation) # stop growth

                    # try one more time next cycle
                    if not short:
                        self.check_validity_filipod(constellation,\
                            new_pos,close_enough,error,ID,\
                            somaID,True)
                    return
                
                try:
                    new_fronts = self.add_branch(constellation,points,\
                                                swc_type=12)
                    # filipod should stop growth
                    self.disable(constellation)
                    # success --> assign mode to new branch
                except (CollisionError, GridCompetitionError,\
                            InsideParentError, VolumeError):
                    # new code: retract front that keeps colliding
                    if (constellation.cycle - self.birth) > 4:
                        if (verbose >= 6) or (verbose_cell == somaID):
                            print (somaID,ID,": _check_validity_filipod_extension in mode = ",self.mode,", have points but could not add branch, disabled at", self.end," interfered by swc/name=",col_front.swc_type,"/",col_front.get_soma(constellation,returnID=True),"located at", col_front.orig)
                        self.disable(constellation) # stop growth
                    return
        
            else: #mode 1,5
                try:
                    new_fronts = self.add_branch(constellation,points,\
                                                swc_type=12)
                    # filipod should stop growth
                    self.disable(constellation)
                    # add_branch success --> assign mode to new fronts
                except (CollisionError, GridCompetitionError,\
                            InsideParentError, VolumeError):
                    # new code: retract front that keeps colliding
                    if (constellation.cycle - self.birth) > 4:
                        self.disable(constellation) # stop growth
                    return

            for f in new_fronts:
                f.mode = self.mode
            return

        else: # no point was given from solve_collision
            
            if (constellation.cycle - self.birth) > 2: # repeated failures
                if self.mode == 3:
                    ref_front = target_front
                    center = Point(target_front.orig.x,target_front.orig.y,self.end.z) + radial_step
                    circle_radius = BGproc_radi + f_radi + close_enough
                    center2 = Point(target_front.orig.x,target_front.orig.y,self.end.z) + (radial_step/3)
                    circle_radius2 = BGproc_radi + f_radi + close_enough + 0.1
                    self.check_validity_2(constellation,new_pos,ID,somaID,col_front,ref_front,center,circle_radius,center2,circle_radius2)
                    return
                
                    if ((verbose >= 6) or (verbose_cell == somaID)):
                        print (somaID,ID,": _check_validity_filipod in mode = ",self.mode,", no points given, disabled at", self.end," interfered by swc/name=",col_front.swc_type,"/",col_front.get_soma(constellation,returnID=True),"located at", col_front.orig)
                self.disable(constellation)
                return
        
            else:# try one more time next cycle
                if (not short) and ( (self.mode == 2) or \
                            (self.mode == 3) or (self.mode == 4)):
                    self.check_validity_filipod(constellation,\
                            new_pos,close_enough,error,\
                            ID,somaID,True)

                return

        print ("ERROR in check_validity_filipod: reached to the bottom without return")


    def check_validity_soma(self,constellation,new_pos,close_enough,\
                            error,ID,somaID,short=False):
        col_front = error.collider
        target_front = constellation.front_by_id(self.target_ID)
    
        if self.loop_lim >= max_loop:
            if ((verbose >= 6) and (self.orig.z > PCL_z)) or (verbose_cell == somaID):
                print (somaID,ID,": _check_validity_soma in mode = ",self.mode," reached loop limit, disable at", self.orig,"(col_front soma:",col_front.get_soma(constellation,returnID=True),")")
                print ("    interfered by swc/name=",col_front.swc_type,"/",col_front.get_id(),"located at", col_front.orig)
            self.mode = -1
            return
    
        if short: # 2nd call, assign shorter new_pos
            if (self.mode == 2) or (self.mode == 3):
                shorter_new_pos = self.orig + (radial_step/3)
                new_pos = shorter_new_pos
            if self.mode == 4:
                shorter_new_pos = self.end + (ag_step/3)
                new_pos = shorter_new_pos
    
        try:
            new_front = self.add_child(constellation,new_pos,\
                            radius=f_radi,swc_type=12,branch_name="filipod")
            return # success
        except CollisionError: # no attempt to correct error
            pass # go to the next code
        except (GridCompetitionError, InsideParentError, VolumeError): # no attempt to correct error
            pass

        points = self.solve_collision(constellation,new_pos,error)
        
        if points: # a list of points
            if (self.mode == 2) or (self.mode == 3) or (self.mode == 4):
                failed = False
                if points[-1].z > self.end.z:
                    failed = True
            
                if failed:
                    if not short:
                        # call again with shorter_new_pos
                        self.check_validity_soma(constellation,new_pos,\
                                        close_enough,error,ID,somaID,short=True)
                    else:
                        self.loop_lim += 1
                        return
                try:
                    new_fronts = self.add_branch(constellation,points,swc_type=12)
                    # add_branch success --> assign mode to new fronts
                except (CollisionError, GridCompetitionError,\
                            InsideParentError, VolumeError):
                    self.loop_lim += 1
                    return
            else: # mode 1,5
                try:
                    new_fronts = self.add_branch(constellation,points,swc_type=12)
                    # add_branch success --> assign mode to new fronts
                except (CollisionError, GridCompetitionError,\
                            InsideParentError, VolumeError):
                    self.loop_lim += 1
                    return

            for f in new_fronts:
                f.mode = self.mode
            return

        else: # no point was given from solve_collision
            if (not short) and ((self.mode == 2) or \
                            (self.mode == 3) or (self.mode == 4)):
                self.check_validity_soma(constellation,new_pos,\
                                close_enough,error,ID,somaID,short=True)
            else:
                self.loop_lim += 1
                return


    # return success (boolean), random is a booolean: add noise to new_pos
    def rootaxon_initiation(self,constellation,target_point,\
                                new_pos,direction,random,ID,somaID):
        
        if random: # decide the x direction away from target BG proc, give random direction
            y_rand = (1.0 - 0.3) * np.random.rand() + 0.3 # 0.3 < y_rand < 1.0
            x_rand = np.random.rand()
            z_rand = (1.0 - 0.3) * np.random.rand() + 0.3

            if target_point.x - self.orig.x < 0:
                new_pos = new_pos + Point(x_rand,y_rand,z_rand)
            else:
                new_pos = new_pos + Point(-x_rand,y_rand,z_rand)

        try:
            new_front = self.add_child(constellation,new_pos,radius=f_radi,\
                                    swc_type=2,branch_name="rootaxon")
            new_front.set_status1() # flag status1 = True
            self.root_pos = new_pos
            # called by soma: do not disable
            return True # success

        except (CollisionError, GridCompetitionError, InsideParentError, VolumeError) as error: # no attempt to correct error
            return False


    def check_validity_PF(self,constellation,new_pos,close_enough,error,ID,somaID):
        col_front = error.collider
        col_front_name = col_front.get_neuron_name(constellation)

        target_front = constellation.front_by_id(self.target_ID)
        target_point = Point(target_front.end.x, target_front.end.y, self.orig.z)
        x_dis = target_point.x - self.orig.x
        soma = self.get_soma(constellation)

        # not far enough from BG proc area
        if ((x_dis < 0) and (self.end.x < soma.root_pos.x)) or ((x_dis > 0) and (self.end.x > soma.root_pos.x )):

            if self.swc_type == 100:
                if x_dis < 0:
                    center = Point(soma.root_pos.x + 0.2, self.end.y - 1.0, self.end.z)
                else:
                    center = Point(soma.root_pos.x - 0.2, self.end.y - 1.0, self.end.z)
            else: #self.swc_type == 101
                if x_dis < 0:
                    center = Point(soma.root_pos.x + 0.2, self.end.y + 1.0, self.end.z)
                else:
                    center = Point(soma.root_pos.x - 0.2, self.end.y + 1.0, self.end.z)
            circle_radius = 0.2
            center2 = center
            circle_radius2 = circle_radius + 0.2

        elif col_front_name.startswith('bergmann') or col_front_name.startswith('pc'):
            z_offset = 0.1
            
            if self.swc_type == 100:
                pf_dir = 1.0
            else:
                pf_dir = -1.0
            center = Point(self.end.x, self.end.y + pf_dir, self.end.z)
            circle_radius = (col_front.radius + self.radius) * 1.1
            
            center2 = center
            circle_radius2 = circle_radius + self.radius#1.0
        
        elif (col_front.swc_type == 2) or (col_front.swc_type == 12) or (col_front.swc_type == 1):
            # assuming collision with vertically oriented granule cell front
            center = Point(self.end.x, col_front.orig.y, self.end.z)
            circle_radius = (col_front.radius + self.radius)*1.1
            
            center2 = center
            circle_radius2 = circle_radius + self.radius#0.2
        
        elif (col_front.swc_type == 100) or (col_front.swc_type == 101):
            if self.swc_type == 100:
                if col_front.swc_type == 100:
                    if col_front.end.y < self.end.y:
                        center = col_front.end
                    else:
                        center = Point(col_front.end.x, self.end.y, col_front.end.z)
                else:
                    y_distance = self.end.y - col_front.end.y
                    if y_distance <= 0.1:
                        center = col_front.orig
                    else:
                        center = col_front.end
            else:
                if col_front.swc_type == 101:
                    if col_front.end.y > self.end.y:
                        center = col_front.end
                    else:
                        y = self.end.y - col_front.end.y
                        center = Point(col_front.end.x, col_front.end.y + y, col_front.end.z)
                else:
                    y_distance = col_front.end.y - self.end.y
                    if y_distance <= 0.1:
                        center = col_front.orig
                    else:
                        center = col_front.end
            circle_radius = (col_front.radius + self.radius)*1.1
            center2 = center
            circle_radius2 = circle_radius + self.radius

        else:
            print ("ERROR in check_validity_PF: interfered by unexpected swc object, col_swc=", col_front.swc_type)
        self.check_validity_2(constellation,new_pos,ID,somaID,col_front,col_front,center,circle_radius,center2,circle_radius2) # 2nd col_front is the same value as ref_front
        return


    # only for PFs and filipod in mode3
    def check_validity_2(self,constellation,new_pos,ID,somaID,\
                            col_front,ref_front,center,circle_radius,center2,circle_radius2):
        
        for further in range(2):
            if further == 0:
                n = (2*np.pi*circle_radius)/0.2
                num_points = int(round(n))

            else:
                center = center2
                circle_radius = circle_radius2
                n2 = (2*np.pi*circle_radius)/0.2
                num_points = int(round(n2))

            if (self.swc_type != 100) and (self.swc_type != 101) and ((self.swc_type != 12)and(self.mode != 3)):
                print ("ERROR in After_check_validity2_, unexpected mode or swc_type:  mode=",self.mode,",swc=",self.swc_type)
                return

            count = 0
            list_made = False
            while True: # endless loop
                count += 1
                if count > 50:#200:
                    if (verbose >= 6) or (verbose_cell == somaID):
                        print (somaID,ID,"__check_validity_2_: killing the while loop,swc/end =",self.swc_type,self.end,", col_front_ID/swc/orig =",col_front,col_front.swc_type,col_front.orig)
                    self.disable(constellation)
                    return
                try:
                    new_front = self.add_child(constellation,new_pos)
                    self.disable(constellation)
                    return # done
                except CollisionError as error: # attempt to correct error
                    if list_made: # made a point_list
                        if point_list: # entries remaining
                            new_pos = point_list.pop(0)
                            continue # while True
                        else: # emptied the list
                            if further == 0:
                                break # go on to the next further loop
                            else: # failed
                                if (verbose >= 6) or (verbose_cell == somaID):
                                    print (somaID,ID,": _check_validity2_ swc & mode = ",self.swc_type,self.mode,", could not find points, TRMINATE at", self.end," interfered by swc/ID = ",col_front.swc_type," / ",col_front.get_soma(constellation,returnID=True),"located at", col_front.orig ,"radius 1 =", circle_radius,", radius 2 =", circle_radius2,"(selfID =",self.get_id(),")")
                                self.disable(constellation)
                                return
                    else: # no list made yet
                        pass # drop down to make a new point_list
                except (GridCompetitionError, InsideParentError,VolumeError):
                    pass # no attempt to correct error

                #make point_list
                point_list = []
                num = 0
                num_point_memory = num_points
                point_counts = 0
                ncount = 0
                while num < num_point_memory:
                    ncount += 1
                    if ncount % 100 == 0:
                        print ("while num loop",ncount)
                    nums = num_points*1.5
                    num_samples = int(round(nums))
                    points = ref_front.alternate_locations(center,circle_radius,num_samples)
                    sorted_points = sorted(points, key=lambda e:(e-self.end).length())
                    if (self.swc_type == 100) or (self.swc_type == 101):
                        if len(sorted_points) > 0:
                            del sorted_points[0]

                    if len(sorted_points) == 0: #no alternative points found
                        if further == 0:
                            break # go on to the next further = 1 loop
                        else:
                            if (verbose >= 6) or (verbose_cell == somaID):
                                print (somaID,ID," : __check_validity_2_, no alternative points found ,self.swc/mode/pos = ",self.swc_type,"/",self.mode,"/",self.end,", interfered by swc/pos/ID = ",col_front.swc_type,"/",col_front.orig,"/",col_front.get_soma(constellation,returnID=True)," // radius 1/radius 2 =",circle_radius,"/",circle_radius2,"(selfID =",self.get_id(),")")
                            self.disable(constellation)
                            return
                
                    if (self.swc_type == 100) or (self.swc_type == 101):
                        target_front = constellation.front_by_id(self.target_ID)
                        target_point = Point(target_front.end.x, target_front.end.y, self.orig.z)
                        x_dis = target_point.x - self.orig.x
                        soma = self.get_soma(constellation)
                    for p in sorted_points:
                        if p == self.end:
                            continue
                        if (self.swc_type == 100) and (p.y >= self.end.y):
                            continue
                        if (self.swc_type == 100) and (col_front.swc_type == 7) and (p.y >= col_front.orig.y):
                            continue
                        if (self.swc_type == 101) and (p.y <= self.end.y):
                            continue
                        if (self.swc_type == 101) and (col_front.swc_type == 7) and (p.y <= col_front.orig.y):
                            continue
                        if ((self.swc_type == 100) or (self.swc_type == 101)) and (x_dis < 0) and (p.x < soma.root_pos.x):
                            continue
                        if ((self.swc_type == 100) or (self.swc_type == 101)) and (x_dis >= 0) and (p.x > soma.root_pos.x):
                            continue
                        if p in point_list:
                            continue
                        point_list.append(p)
                        num +=1

                    point_counts +=1
                    if point_counts > 3: # less points given for the point list
                        num_point_memory = num

                #for safety
                if num == 0:
                    if further == 0:
                        break # go on to the next further = 1 loop
                    else:
                        if (verbose >= 6) or (verbose_cell == somaID):
                            print (somaID,ID," : _check_validity2_ loop2, self swc/mode/self = ",self.swc_type,self.mode,self.end,", 0 alternative_points found")
                            print("      interfered by swc/ID =",col_front.swc_type,"/", col_front.get_soma(constellation,returnID=True))
                        self.disable(constellation)
                        return
                list_made = True

        print("ERROR! in check_validity_2: reached the bottom without returns")


class InvadingPF(SynFront):
                
    def manage_front(self,constellation):
        ID = self.get_id()
        somaID = self.get_soma(constellation,returnID=True)
        
        if self.swc_type == 1:
            # set initial firing rate
            neuron = self.get_neuron(constellation)
            rate = np.random.random() * NEW_FR
            neuron.set_firing_rate(constellation,rate)
            
            if self.orig.y >= 155: # = y_epos_min in admin
                new_pos = self.end - pf_step
                swc = 100
            elif self.orig.y <= 0: # = y_eneg_max in admin
                new_pos = self.end + pf_step
                swc = 101
            else:
                print ("ERROR in InvadingPF:",self.get_id(),"initiated in a wrong place",self.orig)
                self.disable(constellation)
                return

            try:
                new_front = self.add_child(constellation,new_pos,radius=f_radi,swc_type=swc)
                self.disable(constellation)
                return
            except VolumeError:
                # this error is expected: reached volume border -> stop growth
                self.disable(constellation)
                return # done for this cycle
            except (CollisionError, GridCompetitionError, InsideParentError):
                self.disable(constellation)
                return # done for this cycle

        elif (self.swc_type == 100) or (self.swc_type == 101):
            if self.swc_type == 100:
                new_pos = self.end - pf_step
            if self.swc_type == 101:
                new_pos = self.end + pf_step

            try:
                new_front = self.add_child(constellation,new_pos)
                self.disable(constellation)
                return
            except CollisionError as error: # attempt to correct error
                self.check_validity_InvadingPF(constellation,new_pos,error,ID,somaID)
                return
            except (VolumeError):
                self.disable(constellation) # stop growing
                return
            except (GridCompetitionError, InsideParentError):
                # no attempt to correct error
                self.disable(constellation)
                return
    
        print(ID,"ERROR! in InvadingPF manage_front: at the bottom without returns")


    def check_validity_InvadingPF(self,constellation,new_pos,error,ID,somaID):
        col_front = error.collider
        col_front_name = col_front.get_neuron_name(constellation)
        

        if col_front_name.startswith('bergmann') or col_front_name.startswith('pc'):
            z_offset = 0.1
            if self.swc_type == 100:
                pf_dir = 1.0
            else:
                pf_dir = -1.0
            center = Point(self.end.x, self.end.y + pf_dir, self.end.z)
            circle_radius = (col_front.radius + self.radius) * 1.1
            
            center2 = center
            circle_radius2 = circle_radius + self.radius#1.0

        elif (col_front.swc_type == 2) or (col_front.swc_type == 12) or (col_front.swc_type == 1):
            # assuming collision with vertically oriented granule cell front
            center = Point(self.end.x, col_front.orig.y, self.end.z)
            circle_radius = (col_front.radius + self.radius)*1.1
            
            center2 = center
            circle_radius2 = circle_radius + self.radius

        elif (col_front.swc_type == 100) or (col_front.swc_type == 101):
            if self.swc_type == 100:
                if col_front.swc_type == 100:
                    if col_front.end.y < self.end.y:
                        center = col_front.end
                    else:
                        center = Point(col_front.end.x, self.end.y, col_front.end.z)
                else:
                    y_distance = self.end.y - col_front.end.y
                    if y_distance <= 0.1:
                        center = col_front.orig
                    else:
                        center = col_front.end
            else:
                if col_front.swc_type == 101:
                    if col_front.end.y > self.end.y:
                        center = col_front.end
                    else:
                        y = self.end.y - col_front.end.y
                        center = Point(col_front.end.x, col_front.end.y + y, col_front.end.z)
                else:
                    y_distance = col_front.end.y - self.end.y
                    if y_distance <= 0.1:
                        center = col_front.orig
                    else:
                        center = col_front.end
            circle_radius = (col_front.radius + self.radius)*1.1
            center2 = center
            circle_radius2 = circle_radius + self.radius

        else:
            print ("ERROR in check_validity_InvadingPF: interfered by unexpected swc object, col_swc=", col_front.swc_type)

        self.check_validity_IP2(constellation,new_pos,ID,somaID,col_front,center,circle_radius,center2,circle_radius2)
        return


    def check_validity_IP2(self,constellation,new_pos,ID,somaID,\
                            col_front,center,circle_radius,center2,circle_radius2):
        
        for further in range(2):
            if further == 0:
                n = (2*np.pi*circle_radius)/0.2
                num_points = int(round(n))
            else:
                center = center2
                circle_radius = circle_radius2
                n2 = (2*np.pi*circle_radius)/0.2
                num_points = int(round(n2))

            if (self.swc_type != 100) and (self.swc_type != 101):
                print ("ERROR in After_check_validityIP2_, unexpected mode or swc_type:  swc=",self.swc_type)
                return
            count = 0
            list_made = False
            while True: # endless loop
                count += 1
                if count > 50:#200:
                    if (verbose >= 6) or (verbose_cell == somaID):
                        print ("__check_validity_IP2_: killing the while loop,swc/end =",self.swc_type,self.end,", col_front_ID/swc/orig =",col_front,col_front.swc_type,col_front.orig)
                    self.disable(constellation)
                    return

                try:
                    new_front = self.add_child(constellation,new_pos)
                    self.disable(constellation)
                    return # done
                except CollisionError as error: # attempt to correct error
                    if list_made: # made a point_list
                        if point_list: # entries remaining
                            new_pos = point_list.pop(0)
                            continue # while True
                        else: # emptied the list
                            if further == 0:
                                break # force next of for further loop
                            else: # failed
                                if (verbose >= 6) or (verbose_cell == somaID):
                                    print (somaID,ID,": _check_validityIP2_ swc = ",self.swc_type,", could not find points, TRMINATE at", self.end," interfered by swc/ID = ",col_front.swc_type," / ",col_front.get_soma(constellation,returnID=True),"located at", col_front.orig ,"radius 1 =", circle_radius,", radius 2 =", circle_radius2,"(selfID =",self.get_id(),")")
                                self.disable(constellation)
                                return
                    else: # no list made yet
                        pass
                except (GridCompetitionError, InsideParentError,VolumeError):
                    pass # no attempt to correct error

                # Error condition -> make point_list
                point_list = []
                num = 0
                num_point_memory = num_points
                point_counts = 0
                ncount = 0
                while num < num_point_memory:
                    ncount += 1
                    if ncount % 100 == 0:
                        print ("while num loop",ncount)
                    nums = num_points*1.5
                    num_samples = int(round(nums))
                    points = col_front.alternate_locations(center,circle_radius,num_samples)
                    sorted_points = sorted(points, key=lambda e:(e-self.end).length())
                    if (self.swc_type == 100) or (self.swc_type == 101):
                        if len(sorted_points) > 0:
                            del sorted_points[0]

                    if len(sorted_points) == 0:
                        if further == 0:
                            break # go on to the next further = 1 loop
                        else:
                            if (verbose >= 6) or (verbose_cell == somaID):
                                print (somaID,ID," : __check_validity_IP2_, no alternative points found ,self.swc/pos = ",self.swc_type,"/",self.end,", interfered by swc/pos/ID = ",col_front.swc_type,"/",col_front.orig,"/",col_front.get_soma(constellation,returnID=True)," --> return t // radius 1/radius 2 =",circle_radius,"/",circle_radius2)
                            
                            self.disable(constellation)
                            return

                    for p in sorted_points:
                        if p == self.end:
                            continue
                        if (self.swc_type == 100) and (p.y >= self.end.y):
                            continue
                        if (self.swc_type == 100) and (col_front.swc_type == 7) and (p.y >= col_front.orig.y):
                            continue
                        if (self.swc_type == 101) and (p.y <= self.end.y):
                            continue
                        if (self.swc_type == 101) and (col_front.swc_type == 7) and (p.y <= col_front.orig.y):
                            continue
                        if p in point_list:
                            continue
                        point_list.append(p)
                        num +=1

                    point_counts +=1
                    if point_counts > 3:# less points given for the point list
                        num_point_memory = num

                #for safety
                if num == 0:
                    if further == 0:
                        break # go on to the next further = 1 loop
                    else:
                        if (verbose >= 6) or (verbose_cell == somaID):

                            print (somaID,ID," : _check_validityIP2_ loop2, self swc/self = ",self.swc_type,self.end,", 0 alternative_points found  --> return t")
                            print("      interfered by swc/ID =",col_front.swc_type,"/", col_front.get_soma(constellation,returnID=True))
#
                        self.disable(constellation)
                        return
                list_made = True

        print(ID,"ERROR! in InvadingPF check_validity_IP2: reached at the bottom without return")





if __name__ == '__main__':

    
    start = time.time()
    
    s = int(sys.argv[1])
    fname = "db_files_inDeigo/Note45D_part1_c64_seed" + str(s) + ".db"
    #fname = "db_files/Note43_part1_testRun_" + str(s) + ".db"

    #sim_volume = [[-20., -20., -20.], [180.0,160.0,140.0]]
    sim_volume = [[-20., -160., -20.], [180.0,300.0,140.0]]
    neuron_types = [PurkinjeCell,BergmannGlia,GranuleCell,InvadingPF]
    #15
    #63
    #127
    admin = Admin_agent(63,fname,sim_volume,neuron_types,seed=s,verbose=1,max_neurons=14000,\
                        max_fronts=2000000,max_arcs=100,max_active=40000,max_substrate=1280,\
                        max_synapse=120000,grid_step=10.,grid_extra=200)
    
    admin.importable_db = True # writes additional data to database so that it can be imported
    
    degree_1 = 11 # PC raws deviate by 11 degree from the transverse axis of the folium (Palkovits et al.,1971)
    theta_1 = degree_1 * np.pi / 180
    cos = np.cos(theta_1)
    sec = np.arccos(cos)
    point_list_1 = []
    rhombic_side = 20#18
    rhombic_height = 17#15 rhomboid sides and average height corrsponds to 1.19 (Palkovits et al.,1971): h = 18/1.19 = 15
    for i in range (0,6):
        adj = 9.5 + i*rhombic_side
        hyp = adj * sec
        pos = Point(adj,9.5+(hyp*np.sin(theta_1)),0)
        point_list_1.append(pos)
    degree_2 = 14.5 # deviance from y-axis: 90-11-64.5 = 14.5
    theta_2 = degree_2 * np.pi / 180

    list_of_lists = []
    for i in range(0,len(point_list_1)):
        list_of_lists.append([])
        ref_x = point_list_1[i].x
        ref_y = point_list_1[i].y
        for j in range (0,7):
            height = rhombic_height + j*rhombic_height
            hyp = rhombic_side # as rhomboid (or 35.5[perpendicular dir in cat]/2.4[adjust as mice] = 15)
            pos = Point(ref_x + np.sin(theta_2)*hyp, ref_y + height, 0)
            ref_x = pos.x
            list_of_lists[i].append(pos)

    # make first row of Purkinje cells
    for pos in point_list_1:
        admin.add_neurons(PurkinjeCell,"pc",1,\
                                    [[pos.x,pos.y,-5],[pos.x,pos.y,-2]],6)

    # make columns of Purkinje cells
    for i in range(0,len(list_of_lists)):
        for pos in list_of_lists[i]:
            admin.add_neurons(PurkinjeCell,"pc",1,\
                                    [[pos.x,pos.y,-5],[pos.x,pos.y,-2]],6)


    # store Purkinje cell attributes for import:
    admin.attrib_to_db(PurkinjeCell,['prev_branch','signal','stage'],["int","real","int"],last_only=True)

    rhs_m = rhombic_side/2 # rhombic side middle length
    rhh_m = rhombic_height/2 # rhombic height middle length

    BG_z = -6# swc99.end = 10 when BG_z = 0 (want to make swc99.end = 4 max top of PC soma)

    # BGs around 1st row of Purkinje cells
    for pos in point_list_1:
        # extra BGs around the 1st PC in the 1st row
        if pos == point_list_1[0]:
            admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x-rhs_m,pos.y+rhh_m,BG_z],3)
            # BG exactly next by the corner PC
            admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x-rhs_m,pos.y,BG_z],3)
        # BGs outside of PC set and inside of 1st row
        admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x-rhs_m,pos.y-rhh_m,BG_z],3)
        admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x+rhs_m,pos.y+rhh_m,BG_z],3)
        # BGs exactly next by 1st row of PCs
        admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x+rhs_m,pos.y,BG_z],3)
        # extra BG next by the last PC in the 1st row outside of the PC set (corner)
        if pos == point_list_1[-1]:
            admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x+rhs_m,pos.y-rhh_m,BG_z],3)

    # BGs aligned to columns of PCs
    for i in range(0,len(list_of_lists)):
        if i == 0: # outer line of BGs next by the 1st column of PCs
            for pos in list_of_lists[i]:
                admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x-rhs_m,pos.y+rhh_m,BG_z],3)
        for pos in list_of_lists[i]:
            admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x+rhs_m,pos.y+rhh_m,BG_z],3)

    # BGs next by PC columns
    for i in range(0,len(list_of_lists)):
        if i == 0: # outer line of BGs exactly next by 1st column of PCs
            for pos in list_of_lists[i]:
                admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x-rhs_m,pos.y,BG_z],3)
        for pos in list_of_lists[i]:
            admin.add_neurons(BergmannGlia,"bergmann",1,[pos.x+rhs_m,pos.y,BG_z],3)

    try:
        admin.simulation_loop(5)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()


    ML_base = 15# PC pos + PC radi, sync with ML_base in global variable list
    # ML thickness = 121µm in P15 (Faust,2003)
    x_min = -15#-20 + 20
    x_max = 140#175#75#80#180 - 20
    y_min = 0#-15#-5#-20 + 15
    y_max = 40#75#25#160 - 15

    x_min2 = -5
    x_max2 = 155#160
    y_min2 = 40
    y_max2 = 90

    x_min3 = 5
    x_max3 = 175
    y_min3 = 90
    y_max3 = 145

    x_e_max = 160
    x_e_min = -5
    y_epos_min = 155
    y_epos_max = 290
    y_eneg_min = -150
    y_eneg_max = 0

    soma_radi = 0.1 # sync with f_radi

    print ("\nTeam Alfa____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p1",84,\
                [[x_min,y_min,ML_base],[x_max,y_max,ML_base+10]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p1",84,\
                [[x_min2,y_min2,ML_base],[x_max2,y_max2,ML_base+10]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p1",84,\
                [[x_min3,y_min3,ML_base],[x_max3,y_max3,ML_base+10]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p1_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+10]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p1_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+10]],soma_radi)

    # store Granule cell attributes for import: this will apply to ALL Granule cells
    admin.attrib_to_db(GranuleCell,['mode','target_ID',\
                       'loop_lim','root_pos','nograndchildD_loop'],\
                       ["int","id",\
                       "int","point","int"],last_only=True)
    # save neuron firing_rate at end of simulation
    admin.attrib_to_db(GranuleCell,"firing_rate","real",object=Neuron,last_only=True)

    # save neuron firing_rate at end of simulation
    admin.attrib_to_db(InvadingPF,"firing_rate","real",object=Neuron,last_only=True)

    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam Bravo____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p2",84,\
                [[x_min,y_min,ML_base+10],[x_max,y_max,ML_base+20]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p2",84,\
                [[x_min2,y_min2,ML_base+10],[x_max2,y_max2,ML_base+20]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p2",84,\
                [[x_min3,y_min3,ML_base+10],[x_max3,y_max3,ML_base+20]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p2_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+20]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p2_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+20]],soma_radi)

    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam Charlie____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p3",84,\
                [[x_min,y_min,ML_base+20],[x_max,y_max,ML_base+30]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p3",84,\
                [[x_min2,y_min2,ML_base+20],[x_max2,y_max2,ML_base+30]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p3",84,\
                [[x_min3,y_min3,ML_base+20],[x_max3,y_max3,ML_base+30]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p3_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+30]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p3_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+30]],soma_radi)

    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam Delta____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p4",84,\
                [[x_min,y_min,ML_base+30],[x_max,y_max,ML_base+40]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p4",84,\
                [[x_min2,y_min2,ML_base+30],[x_max2,y_max2,ML_base+40]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p4",84,\
                [[x_min3,y_min3,ML_base+30],[x_max3,y_max3,ML_base+40]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p4_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+40]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p4_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+40]],soma_radi)

    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam Echo____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p5",84,\
                [[x_min,y_min,ML_base+40],[x_max,y_max,ML_base+50]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p5",84,\
                [[x_min2,y_min2,ML_base+40],[x_max2,y_max2,ML_base+50]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p5",84,\
                [[x_min3,y_min3,ML_base+40],[x_max3,y_max3,ML_base+50]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p5_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+50]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p5_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+50]],soma_radi)

    try:
        admin.simulation_loop(10) #55
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam Foxtrot____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p6",84,\
                [[x_min,y_min,ML_base+50],[x_max,y_max,ML_base+60]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p6",84,\
                [[x_min2,y_min2,ML_base+50],[x_max2,y_max2,ML_base+60]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p6",84,\
                [[x_min3,y_min3,ML_base+50],[x_max3,y_max3,ML_base+60]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p6_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+60]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p6_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+60]],soma_radi)

    try:
        admin.simulation_loop(10) #65
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()

    print ("\nTeam GATO____________________________________________________")
    admin.add_neurons(GranuleCell,"granule_p7",84,\
                [[x_min,y_min,ML_base+60],[x_max,y_max,ML_base+70]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p7",84,\
                [[x_min2,y_min2,ML_base+60],[x_max2,y_max2,ML_base+70]],soma_radi,migrating=True)
    admin.add_neurons(GranuleCell,"granule_p7",84,\
                [[x_min3,y_min3,ML_base+60],[x_max3,y_max3,ML_base+70]],soma_radi,migrating=True)

    admin.add_neurons(InvadingPF,"granule_p7_exp",429,\
                [[x_e_min,y_epos_min,ML_base],[x_e_max,y_epos_max,ML_base+70]],soma_radi)
    admin.add_neurons(InvadingPF,"granule_p7_exn",429,\
                [[x_e_min,y_eneg_min,ML_base],[x_e_max,y_eneg_max,ML_base+70]],soma_radi)

    try:
        admin.simulation_loop(5) #70
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()


    # clean up
    print ("Cerebellum simulation:",s,time.time()-start,"seconds")
    admin.destruction()





