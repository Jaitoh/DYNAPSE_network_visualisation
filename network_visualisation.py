'''
A function helps graphically visualize the network connections.
It can be a alternative option to net_gen.print_network() -> draw_network(net_gen, title="Network", store_path = "./data/")
'''

'''
Author: Wenjie He
Date:   June 2021
Ver:    1.0
'''


import sys
sys.path.insert(1, '/home/class_NI2021/ctxctl_contrib')

import samna.dynapse1 as dyn1
import io
import NetworkGenerator as n
import Dynapse1Utils as ut
from NetworkGenerator import Neuron
import re
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class draw_network():
    """ draw the network with the output of net_gen.print_network()
    """

    def __init__(self, net_gen, title="Network", store_path = "./data/"):
        # get the network output
        with io.StringIO() as buf, redirect_stdout(buf):
            net_gen.print_network()
            output = buf.getvalue()
        # save the output 
        f = open(store_path+"network.txt",'w')
        f.write(output)  # save the network content
        
        with open(store_path+"network.txt") as f:
            contents = f.readlines()
        NN = []
        for line in contents:
            if "[('" in line:
                NN.append(line)

        network_DF = self.generateDF(NN)
        self.draw(network_DF, title=title, store_path = store_path)

    def generateDF(self, NN):
        """ transform collected print_out into dataframe
        """
        # transform into dataframe
        network_df = pd.DataFrame(columns=("Source","Target","Synapse","Number"))
        Targets = []
        for i in range(len(NN)):
            Sources = []
            Target = re.search("(.*):", NN[i]).group(1)
            Targets.append(Target)
            Sources_raw = re.search("\[(.*)\]", NN[i]).group(1).split(")")
            
            for j in range(len(Sources_raw)-1):
                Source = re.search("\((.*),", Sources_raw[j]).group(1)[1:-1]
                Synapse = re.search(", (.*)", Sources_raw[j]).group(1)[1:-1]
                if Source not in Sources: # a new connection
                    Sources.append(Source)
                    count = 1
                    network_df = network_df.append({'Source': Source, 'Target': Target, 'Synapse': Synapse, 'Number': count}, ignore_index=True)
                else: # an existing connection
                    current_loc = (network_df.Source==Source) & (network_df.Target==Target)
                    network_df.loc[current_loc , "Number"] = network_df.loc[current_loc].Number + 1
        return network_df

    def draw(self, network_DF, title="Network", store_path = "./data/"):
        
        network_DF = network_DF.sort_values(by=["Source", "Target", "Synapse", "Number"])
        
        self.imageTitle = title + "\nAMPA_red NMDA_orange GABA_A_blue GABA_B_green"
        self.imageSave = title
        
        self.showSpikeGen = True # display the connection between spikegen and neurons

        self.setParameters() # set plotting parameters

        figure, axes = plt.subplots()
        plt.rcParams["figure.figsize"] = (6,6)
        plt.rcParams['savefig.dpi'] = 400 
        plt.rcParams['figure.dpi'] = 400 

        neuron_allocation, hardware_layout = self.reshapeDataframe(network_DF)

        self.drawNeurons(neuron_allocation, hardware_layout, axes)
        self.drawChipCore(hardware_layout, axes)
        self.drawConnections(network_DF, neuron_allocation, axes)

        self.saveImage(hardware_layout, figure, axes, store_path)

    def setParameters(self):
        # plotting parameters
        self.radius = 10 # size of the neuron
        self.maxNeuron = 8 # max number of neuron per line
        self.radius_port = 0.5
        self.fontsize =  5 # 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large
        self.distance = 6.5*self.radius # should be bigger than 6 times of raiuds

        # line styles
        self.lw_neuron = 1
        self.lw_core = 0.5
        self.lw_chip = 0.5
        self.lw_AMPA = 0.5
        self.lw_NMDA = 0.5
        self.lw_GABA_A = 0.5
        self.lw_GABA_B = 0.5
        self.head_width=5
        self.head_length=5

        self.alpha = 0.2
        self.ls_AMPA = "-"
        self.ls_NMDA = "-"
        self.ls_GABA_A = "-"
        self.ls_GABA_B = "-"
        self.ls_spike_gen = '-'

        # color settings
        self.color_neuron = "k"
        self.color_core = "tab:blue"
        self.color_chip = "tab:green"
        self.color_AMPA = "tab:red"
        self.color_NMDA = "tab:orange"
        self.color_GABA_A = "tab:blue"
        self.color_GABA_B = "tab:green"
        self.color_num_conn = "k"

    def reshapeDataframe(self, network_DF):
    # list all neurons used in the network
        all_neurons = np.unique([network_DF["Source"].tolist(), network_DF["Target"].tolist()])
        # print("all_neurons:", all_neurons)

        # reform string into dataframe for neuron allocations
        neuron_allocation = pd.DataFrame(columns=["chip_id", "core_id", "neuron_id", "spike_gen"])
        
        for neuron in all_neurons:
            chip_id = re.search('C(.*)c', neuron).group(1) # search chip id
            # print("chip_id",chip_id)
            core_id = re.search('c(.*)[s,n]', neuron).group(1) # search core id
            # print("core_id",core_id)
            neuron_id = re.search('[s,n](.*)', neuron).group(1) # search neuron id
            # print("neuron_id",neuron_id)
            if re.search('c[0-9]+(.)[0-9]+', neuron).group(1) =="s": # check if is spike_gen
                spike_gen = 1
            else:
                spike_gen = 0
            # count the input connection and the output connection number
            num_connection_in = np.sum(network_DF.loc[network_DF.Target==neuron].Number)
            num_connection_out = np.sum(network_DF.loc[network_DF.Source==neuron].Number)
            df2 = pd.DataFrame(np.array([[chip_id, core_id, neuron_id, spike_gen, num_connection_in, num_connection_out]]), columns = ["chip_id", "core_id", "neuron_id", "spike_gen", "connection_in", "connection_out"])
            neuron_allocation = neuron_allocation.append(df2).reset_index(drop=True)
        neuron_allocation = neuron_allocation.astype(int)
        neuron_allocation = neuron_allocation.sort_values(by=['chip_id', 'core_id', 'neuron_id', 'spike_gen'])
        # add geometric point for further plotting
        neuron_allocation["x_in"], neuron_allocation["y_in"], neuron_allocation["x_out"], neuron_allocation["y_out"] = 0, 0, 0, 0
        # print(neuron_allocation.tail())

        # how many cores in each chip and how many neurons in each core
        hardware_layout = pd.DataFrame(columns=["chip_id", "core_id", "num_neuron"])
        # find the used chips
        used_chips = np.unique(neuron_allocation.chip_id.to_list())
        # find the number of cores used in each chip
        for chip in used_chips:
            neurons_in_chip = neuron_allocation.loc[neuron_allocation["chip_id"]==chip]
            used_cores = np.unique(neurons_in_chip.core_id.tolist())
            for core in used_cores:
                neurons_in_core = neurons_in_chip.loc[neurons_in_chip["core_id"]==core]
                num_neuron = len(neurons_in_core)
                hardware_layout = hardware_layout.append(pd.DataFrame([[chip, core, num_neuron]], columns=["chip_id", "core_id", "num_neuron"]))
        hardware_layout = hardware_layout.astype(int)
        hardware_layout = hardware_layout.sort_values(by=['chip_id', 'core_id', "num_neuron"])
        hardware_layout = hardware_layout.reset_index(drop=True)
        hardware_layout["num_col_draw"] = np.ceil(hardware_layout["num_neuron"]/self.maxNeuron).astype(int)
        # print("hardware_layout:", hardware_layout)
        return neuron_allocation, hardware_layout

    def drawNeurons(self, neuron_allocation, hardware_layout, axes):
        print("\nstart drawing the network...")

        circles = []
        x_center = self.distance
        self.x_center_init = x_center
        self.y_center_init = 4*self.radius
        print("generating neurons...")
        for i in range(len(hardware_layout)):
            num_neurons_in_core = hardware_layout.loc[i].num_neuron
            current_chip = hardware_layout.loc[i].chip_id # current chip id
            current_core = hardware_layout.loc[i].core_id # current core id
            y_center = self.y_center_init
            # neuron list of the current chip and core
            current_ChipCore = (neuron_allocation.chip_id==current_chip) & (neuron_allocation.core_id==current_core)
            neuron_id_list = neuron_allocation.loc[current_ChipCore].neuron_id.to_list()
            neuron_num = len(neuron_id_list)
            spike_gen_list = neuron_allocation.loc[current_ChipCore].spike_gen.to_list()
            conn_in = neuron_allocation.loc[current_ChipCore].connection_in.to_list()
            conn_out = neuron_allocation.loc[current_ChipCore].connection_out.to_list()
            neuron_info = zip(*(neuron_id_list, spike_gen_list, conn_in, conn_out))
            count_neuron = 0
            for neuron_id, spike_gen, conn_in, conn_out in neuron_info:
                # draw a circleX
                circles.append(plt.Circle((x_center, y_center), self.radius, fill=False, edgecolor=self.color_neuron, lw=self.lw_neuron))
                if spike_gen: 
                    neuron_display = str(neuron_id)+"s"
                else:
                    neuron_display = str(neuron_id)
                # print text inside circle
                # print(x_center, y_center-radius/4)
                axes.annotate(neuron_display, (x_center, y_center-2*self.radius), fontsize=self.fontsize, horizontalalignment='center')
                # plt.text(x_center, y_center-radius/4, neuron_display, color=self.color_neuron, horizontalalignment='center') 
                # add in and out info of the neuron
                x_in = x_center - self.radius
                y_in = y_center
                x_out = x_center + self.radius
                y_out = y_center
                current_neuron_loc = (neuron_allocation.chip_id==current_chip) & (neuron_allocation.core_id==current_core) & (neuron_allocation.neuron_id==neuron_id) & (neuron_allocation.spike_gen==spike_gen)
                neuron_allocation.loc[current_neuron_loc, "x_in"] = x_in
                neuron_allocation.loc[current_neuron_loc, "y_in"] = y_in
                neuron_allocation.loc[current_neuron_loc, "x_out"] = x_out
                neuron_allocation.loc[current_neuron_loc, "y_out"] = y_out
                circles.append(plt.Circle((x_in, y_in), self.radius_port, fill=True, edgecolor=None, color=self.color_neuron))
                circles.append(plt.Circle((x_out, y_out), self.radius_port, fill=True, edgecolor=None, color=self.color_neuron)) # draw small point at the in/out port
                # display the in/out number of connections
                axes.annotate(conn_in, (x_center-5*self.radius/4, y_center-self.radius/4), fontsize=self.fontsize/2, horizontalalignment='right', color=self.color_num_conn)
                axes.annotate(conn_out, (x_center+5*self.radius/4, y_center-self.radius/4), fontsize=self.fontsize/2, horizontalalignment='left', color=self.color_num_conn)
                count_neuron += 1
                if count_neuron%self.maxNeuron==0 and count_neuron!= neuron_num: # change into a new line to draw the neuron when reaching self.maxNeuron per col (when not the last neuron of the current core)
                    y_center = self.y_center_init
                    x_center += self.distance
                else:
                    y_center += self.distance
            x_center += self.distance
        # draw neuron, core and chip
        for circle in circles:
            axes.add_patch(circle)

    def drawChipCore(self, hardware_layout, axes):
        print("generating cores...")
        # draw core
        rec_coreW = (self.maxNeuron-1)*self.distance + 4*self.radius
        rec_cores = []
        x_center = self.x_center_init - self.radius*2
        y_center = self.y_center_init - self.radius*2
        core_id = hardware_layout.core_id.to_list()
        num_col = hardware_layout.num_col_draw.to_list()
        core_info = zip(*(core_id, num_col))
        for core_id, num_col in core_info:
            rec_coreL = self.distance*(num_col-1) + 4*self.radius
            rec_cores.append(patches.Rectangle((x_center, y_center), rec_coreL, rec_coreW, linewidth=self.lw_core, edgecolor=self.color_core, facecolor='none'))
            plt.text(x_center+rec_coreL/2, y_center-self.radius, "core "+str(core_id), horizontalalignment='center', color = self.color_core, fontsize = self.fontsize)
            x_center += rec_coreL + self.distance - 4*self.radius

        print("generating chips...")
        # draw chip
        rec_chipW = (self.maxNeuron-1)*self.distance+6*self.radius
        rec_chips = []
        x_center = self.x_center_init - self.radius*3
        y_center = self.y_center_init - self.radius*3
        for chip_id in np.unique(hardware_layout.chip_id): # determin the number of chip
            num_col = np.sum(hardware_layout.loc[hardware_layout.chip_id==chip_id].num_col_draw)
            rec_chipL = (num_col-1)*self.distance+6*self.radius
            rec_chips.append(patches.Rectangle((x_center, y_center), rec_chipL, rec_chipW, linewidth=self.lw_chip, edgecolor=self.color_chip, facecolor='none'))
            plt.text(x_center+rec_chipL/2, y_center-self.radius, "chip "+str(chip_id), horizontalalignment='center', color = self.color_chip, fontsize=self.fontsize)
            x_center += rec_chipL + self.distance - 6*self.radius

        for rec_core in rec_cores:
            axes.add_patch(rec_core)

        for rec_chip in rec_chips:
            axes.add_patch(rec_chip)

    def drawConnections(self, network_DF, neuron_allocation, axes):
        print("building connections...")
        # draw arrow
        for i in range(len(network_DF)):
            # parse source
            source = network_DF.loc[i].Source
            source_chip = int(re.search('C(.*)c', source).group(1)) # get chip id
            source_core = int(re.search('c(.*)[s,n]', source).group(1)) # get core id
            source_neuron = int(re.search('[s,n](.*)', source).group(1)) # get neuron id
            # get the starting location
            neuron_info = neuron_allocation.loc[(neuron_allocation.chip_id==source_chip) & (neuron_allocation.core_id==source_core) & (neuron_allocation.neuron_id==source_neuron)].reset_index(drop=True)
            x_start, y_start, is_spikegen = neuron_info.at[0, "x_out"], neuron_info.at[0, "y_out"], neuron_info.at[0, "spike_gen"]
            # parse target
            target = network_DF.loc[i].Target
            target_chip = int(re.search('C(.*)c', target).group(1)) # get chip id
            target_core = int(re.search('c(.*)[s,n]', target).group(1)) # get core id
            target_neuron = int(re.search('[s,n](.*)', target).group(1)) # get neuron id
            # get the ending location
            neuron_info = neuron_allocation.loc[(neuron_allocation.chip_id==target_chip) & (neuron_allocation.core_id==target_core) & (neuron_allocation.neuron_id==target_neuron)].reset_index(drop=True)
            x_end, y_end = neuron_info.at[0, "x_in"], neuron_info.at[0, "y_in"]
            dx, dy = x_end-x_start, y_end-y_start

            # draw arrow
            synapse = network_DF.loc[i].Synapse
            if synapse == "AMPA":
                color_arrow = self.color_AMPA
                ls_arrow = self.ls_AMPA
            if synapse == "NMDA":
                color_arrow = self.color_NMDA
                ls_arrow = self.ls_NMDA
            if synapse == "GABA_A":
                color_arrow = self.color_GABA_A
                ls_arrow = self.ls_GABA_A
            if synapse == "GABA_B":
                color_arrow = self.color_GABA_B
                ls_arrow = self.ls_GABA_B
            if is_spikegen and self.showSpikeGen==False: #if self.showSpikeGen==False, skip drawing current arrow
                continue
            
            axes.arrow(x_start, y_start, dx, dy, color=color_arrow, ls=ls_arrow, alpha=self.alpha, head_width=self.head_width, head_length=self.head_length) # label=synapse
            
            # display the number of connection above the arrow
            num_connection = network_DF.loc[i].Number
            x_text, y_text = (x_start+x_end)/2, (y_start+y_end)/2
            plt.text(x_text, y_text, num_connection, horizontalalignment='center', color=color_arrow, fontsize=self.fontsize)
    
    def saveImage(self, hardware_layout, figure, axes, store_path, showImage=True):
        if self.showSpikeGen==False:
            self.imageTitle+="(spikeGen hidden)"
        plt.title(self.imageTitle)
        ymax = (self.maxNeuron-1)*self.distance + 10*self.radius
        xmax = self.distance*(np.sum(hardware_layout.num_col_draw)+1)
        plt.xlim(-10,xmax)
        plt.ylim(-10,ymax)
        axes.axis("off")
        axes.set_aspect(1)
        if showImage:
            plt.show()
        figure.savefig(store_path+self.imageSave)
        print(f"done \nfigure saved to {store_path}{self.imageSave}.png")
        
def gen_param_group_1core():
    paramGroup = dyn1.Dynapse1ParameterGroup()
    # THR, gain factor of neurons
    paramGroup.param_map["IF_THR_N"].coarse_value = 5
    paramGroup.param_map["IF_THR_N"].fine_value = 80

    # refactory period of neurons
    paramGroup.param_map["IF_RFR_N"].coarse_value = 4
    paramGroup.param_map["IF_RFR_N"].fine_value = 128

    # leakage of neurons
    paramGroup.param_map["IF_TAU1_N"].coarse_value = 4
    paramGroup.param_map["IF_TAU1_N"].fine_value = 80

    # turn off tau2
    paramGroup.param_map["IF_TAU2_N"].coarse_value = 7
    paramGroup.param_map["IF_TAU2_N"].fine_value = 255

    # turn off DC
    paramGroup.param_map["IF_DC_P"].coarse_value = 0
    paramGroup.param_map["IF_DC_P"].fine_value = 0

    # leakage of AMPA
    paramGroup.param_map["NPDPIE_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_F_P"].fine_value = 80

    # gain of AMPA
    paramGroup.param_map["NPDPIE_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_F_P"].fine_value = 80

    # weight of AMPA
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 0

    # leakage of NMDA
    paramGroup.param_map["NPDPIE_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_S_P"].fine_value = 80

    # gain of NMDA
    paramGroup.param_map["NPDPIE_THR_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_S_P"].fine_value = 80

    # weight of NMDA
    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 0

    # leakage of GABA_A (shunting)
    paramGroup.param_map["NPDPII_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_F_P"].fine_value = 80

    # gain of GABA_A (shunting)
    paramGroup.param_map["NPDPII_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_THR_F_P"].fine_value = 80

    # weight of GABA_A (shunting)
    paramGroup.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_INH_F_N"].fine_value = 0

    # leakage of GABA_B
    paramGroup.param_map["NPDPII_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_S_P"].fine_value = 80

    # gain of GABA_B
    paramGroup.param_map["NPDPII_THR_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_THR_S_P"].fine_value = 80

    # weight of GABA_B
    paramGroup.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_INH_S_N"].fine_value = 0

    # other advanced parameters
    paramGroup.param_map["IF_NMDA_N"].coarse_value = 0
    paramGroup.param_map["IF_NMDA_N"].fine_value = 0

    paramGroup.param_map["IF_AHTAU_N"].coarse_value = 4
    paramGroup.param_map["IF_AHTAU_N"].fine_value = 80

    paramGroup.param_map["IF_AHTHR_N"].coarse_value = 0
    paramGroup.param_map["IF_AHTHR_N"].fine_value = 0

    paramGroup.param_map["IF_AHW_P"].coarse_value = 0
    paramGroup.param_map["IF_AHW_P"].fine_value = 0

    paramGroup.param_map["IF_CASC_N"].coarse_value = 0
    paramGroup.param_map["IF_CASC_N"].fine_value = 0

    paramGroup.param_map["PULSE_PWLK_P"].coarse_value = 4
    paramGroup.param_map["PULSE_PWLK_P"].fine_value = 106

    paramGroup.param_map["R2R_P"].coarse_value = 3
    paramGroup.param_map["R2R_P"].fine_value = 85

    paramGroup.param_map["IF_BUF_P"].coarse_value = 3
    paramGroup.param_map["IF_BUF_P"].fine_value = 80

    return paramGroup


if __name__ == '__main__':

    # open DYNAP-SE1 board to get Dynapse1Model
    device_name = "my_dynapse1"
    store = ut.open_dynapse1(device_name, gui=False, sender_port=12305, receiver_port=12306)
    model = getattr(store, device_name)

    # set initial (proper) parameters
    paramGroup = gen_param_group_1core()
    for chip in range(4):
        for core in range(4):
            model.update_parameter_group(paramGroup, chip, core)

    # build a Simple network for test
    net_gen = n.NetworkGenerator()
    schip_id, score_id, sneuron_id = 0, 1, 0
    neuron_pre = Neuron(schip_id, score_id, sneuron_id, True)
    chip_id, core_id, neuron_id1, neuron_id2 = 0, 2, 1, 2
    neuron_pst1 = Neuron(chip_id, core_id, neuron_id1)
    neuron_pst2 = Neuron(chip_id, core_id, neuron_id2)
    net_gen.add_connection(neuron_pre, neuron_pst1, dyn1.Dynapse1SynType.AMPA)
    net_gen.add_connection(neuron_pre, neuron_pst2, dyn1.Dynapse1SynType.AMPA)

    # draw the network
    draw_network(net_gen, title="Network", store_path = "./")

    # close Dynapse1
    ut.close_dynapse1(store, device_name)
