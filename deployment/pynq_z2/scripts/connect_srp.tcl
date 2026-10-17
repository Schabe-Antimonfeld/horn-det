foreach name {srp_accumulator srp_batch} {
    add_files -norecurse [file join $deploy_root rtl ${name}.v]
}
update_compile_order -fileset sources_1
create_bd_cell -type module -reference srp_batch srp_scoring
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 srp_dma
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16}] [get_bd_cells srp_dma]
set_property CONFIG.NUM_MI {4} [get_bd_cells control_bus]
set_property CONFIG.NUM_SI {5} [get_bd_cells memory_bus]
connect_bd_intf_net [get_bd_intf_pins control_bus/M03_AXI] [get_bd_intf_pins srp_dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins srp_dma/M_AXI_MM2S] [get_bd_intf_pins memory_bus/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins srp_dma/M_AXI_S2MM] [get_bd_intf_pins memory_bus/S04_AXI]
connect_bd_intf_net [get_bd_intf_pins srp_dma/M_AXIS_MM2S] [get_bd_intf_pins srp_scoring/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins srp_scoring/M_AXIS] [get_bd_intf_pins srp_dma/S_AXIS_S2MM]
foreach pin {srp_scoring/clk srp_dma/s_axi_lite_aclk srp_dma/m_axi_mm2s_aclk srp_dma/m_axi_s2mm_aclk
             control_bus/M03_ACLK memory_bus/S03_ACLK memory_bus/S04_ACLK} {
    connect_bd_net [get_bd_pins ps/FCLK_CLK0] [get_bd_pins $pin]
}
foreach pin {srp_scoring/resetn srp_dma/axi_resetn} {
    connect_bd_net [get_bd_pins reset_system/peripheral_aresetn] [get_bd_pins $pin]
}
foreach pin {control_bus/M03_ARESETN memory_bus/S03_ARESETN memory_bus/S04_ARESETN} {
    connect_bd_net [get_bd_pins reset_system/interconnect_aresetn] [get_bd_pins $pin]
}
