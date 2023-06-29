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
PC_start_growth = 5
# import starts from cycle 71
wholeCheck_cycle_1 = 30 # 90-65 = 25
wholeCheck_cycle_2 = wholeCheck_cycle_1 + 7

wholeCheck_1 = 250 # threshold of front numbers in a whole neuron to start first retraction process
wholeCheck_2 = 400

first_f_th = 60
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
                self.select_winners(constellation,somaID)
            elif constellation.cycle >= wholeCheck_cycle_2 and self.stage == 2:
                self.select_winners(constellation,somaID)
            elif self.stage == 3:
                self.select_winners(constellation,somaID)
                self.disable(constellation) # done
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


    # count synapses on all dendrite branches and pick winners, retract others
    def select_winners(self,constellation,somaID):
        nname = self.get_neuron_name(constellation)
        
        # caluculate the number of fronts in a whole neuron
        neuron = self.get_neuron(constellation)
        wfront_len = neuron.num_fronts

        if self.stage == 1:
            if wfront_len > wholeCheck_1:
                if (verbose >= 1) or (verbose_cell == somaID):
                    print(nname,"reached whole fronts threshold =",wholeCheck_1,"at",constellation.cycle)
                
                self.stage = 2
                roots = self.get_children(constellation)

                # retract by front threshold
                for root in roots:
                    root_name = root.get_branch_name()
                    efront_len = root.count_descendants(constellation) # get the number of fronts for each branch
            
                    if efront_len < first_f_th:
                        if (verbose >= 1) or (verbose_cell == somaID):
                            print(root_name,"at 1st selection: num front =",efront_len, ", retract")
                        self.retract_branch(constellation,root)
                    else:
                         if (verbose >= 1) or (verbose_cell == somaID):
                            print (root_name,", 1st threshold, EXTEND further with____",efront_len,"fronts")

        elif self.stage == 2: #  final selection
            if wfront_len > wholeCheck_2:
                if (verbose >= 1) or (verbose_cell == somaID):
                    print(nname,"reached whole fronts threshold 2 =",wholeCheck_2,"at",constellation.cycle)
                
                self.stage = 3
                roots = self.get_children(constellation)
                total_front = [0.] * self.num_children
                # generate total_front list of each dendrite
                nd = 0
                for root in roots:
                    efront_len = root.count_descendants(constellation)
                    total_front[nd] = efront_len
                    nd += 1

                nd = 0
                for root in roots:
                    root_name = root.get_branch_name()
                    if total_front[nd] == max(total_front):
                        if (verbose >= 1) or (verbose_cell == somaID):
                            print (root_name,", 2nd threshold, EXTEND further with____",total_front[nd],"fronts")
                    else:
                        if (verbose >= 1) or (verbose_cell == somaID):
                            print(root_name,"at 2nd selection: num front =",total_front[nd], ", retract")
                        self.retract_branch(constellation,root)
                        
                    nd += 1
        

        elif self.stage == 3: #count the final numbers of primary trees
            self.stage == 4
            if (verbose >= 1) or (verbose_cell == somaID):
                print(nname,": final num survivors =",self.num_children)#
        else:
            print ("Error in select_winners PurkinjeCell: unexpected stage =",self.stage)




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



if __name__ == '__main__':

    
    start = time.time()
    
    s = int(sys.argv[1])
    fname = "db_files_inDeigo/Note46D_S0C_S2_1C_400n60_noGC_noImport" + str(s) + ".db"
    #fname = "db_files/Note46_part2_S2_1test_" + str(s) + ".db"
    
    
    sim_volume = [[-20., -160., -20.], [180.0,300.0,140.0]]
    neuron_types = [PurkinjeCell,BergmannGlia]
    #15
    #127
    #63
    admin = Admin_agent(63,fname,sim_volume,neuron_types,verbose=1,max_neurons=14000,\
                        max_fronts=2000000,max_arcs=100,max_active=40000,max_substrate=1280,\
                        max_synapse=120000,grid_step=10.,grid_extra=200)
    
    
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

    
 

    try:
        admin.simulation_loop(5)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()



    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()


    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()


    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()



    try:
        admin.simulation_loop(10)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()




    try:
        admin.simulation_loop(105)
    except Exception as error:
        print ("Failed:",error,s)
        admin.destruction()




    # clean up
    print ("Cerebellum simulation:",s,time.time()-start,"seconds")
    admin.destruction()





