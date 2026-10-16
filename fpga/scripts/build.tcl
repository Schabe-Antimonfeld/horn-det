if {[llength $argv] < 2 || [llength $argv] > 3} {
    error "Usage: build.tcl <new_build_directory> <PYNQ_board_repository_or_dash> ?create|synth|bit?"
}
set output [file normalize [lindex $argv 0]]
set board_repo [lindex $argv 1]
set mode [expr {[llength $argv] == 3 ? [lindex $argv 2] : "create"}]
if {$mode ni {create synth bit}} {error "Mode must be create, synth or bit"}
if {[file exists [file join $output horn_capture.xpr]]} {
    error "Project already exists; choose a new directory or open the existing project in Vivado"
}
if {![string match "2019.2*" [version -short]]} {
    puts "WARNING: This script targets Vivado 2019.2; review any IP migration"
}
if {$mode eq "bit"} {
    foreach name {PDM_INPUT_MIN_NS PDM_INPUT_MAX_NS} {
        if {![info exists ::env($name)] || ![string is double -strict $::env($name)]} {
            error "Set $name from the verified interface min/max budget before implementation"
        }
    }
    if {$::env(PDM_INPUT_MIN_NS) < 0 || $::env(PDM_INPUT_MAX_NS) <= $::env(PDM_INPUT_MIN_NS) ||
        $::env(PDM_INPUT_MAX_NS) >= 156.25} {
        error "Review PDM min/max delays against the minimum half-period of 156.25 ns"
    }
}
if {$board_repo ne "-"} {
    set board_repo [file normalize $board_repo]
    if {![file isdirectory $board_repo]} {error "Board repository does not exist"}
    set_param board.repoPaths [list $board_repo]
}
set boards [lsort -dictionary [get_board_parts -quiet -filter {NAME =~ "*pynq-z2*"}]]
if {[llength $boards] == 0} {
    error "Install the official PYNQ-Z2 board files for Vivado, then provide their repository directory"
}

set root [file normalize [file join [file dirname [info script]] ..]]
create_project horn_capture $output -part xc7z020clg400-1
set_property board_part [lindex $boards end] [current_project]
set_property target_language Verilog [current_project]
foreach name {pdm_rx pdm_decimator axis_pcm pdm_capture} {
    add_files -norecurse [file join $root rtl ${name}.v]
}
add_files -fileset constrs_1 -norecurse [file join $root constraints pynq_z2.xdc]
add_files -fileset sim_1 -norecurse [file join $root sim tb_capture.sv]
set_property top tb_capture [get_filesets sim_1]

create_bd_design capture_system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000}] [get_bd_cells ps]

create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 audio_clock
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {12.288} CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_LOW} \
    CONFIG.MMCM_DIVCLK_DIVIDE {5} CONFIG.MMCM_CLKFBOUT_MULT_F {48.000} \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F {78.125}] [get_bd_cells audio_clock]
set wizard [get_bd_cells audio_clock]
set calculated [expr {100.0 * [get_property CONFIG.MMCM_CLKFBOUT_MULT_F $wizard] / \
    [get_property CONFIG.MMCM_DIVCLK_DIVIDE $wizard] / [get_property CONFIG.MMCM_CLKOUT0_DIVIDE_F $wizard]}]
if {abs($calculated - 12.288) > 0.000001} {error "Clocking Wizard did not retain the required 12.288 MHz configuration"}

foreach name {reset_system reset_audio} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 $name
    set_property CONFIG.C_EXT_RESET_HIGH {0} [get_bd_cells $name]
}
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 zero
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] [get_bd_cells zero]
create_bd_cell -type module -reference pdm_capture capture

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 stream_fifo
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} CONFIG.IS_ACLK_ASYNC {1} CONFIG.FIFO_DEPTH {1024} \
    CONFIG.FIFO_MODE {1}] [get_bd_cells stream_fifo]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 audio_dma
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {0} \
    CONFIG.c_include_s2mm {1} CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_s2mm_data_width {64} CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_s2mm_dre {0} CONFIG.c_s2mm_burst_size {16}] [get_bd_cells audio_dma]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 capture_control
set_property -dict [list CONFIG.C_GPIO_WIDTH {1} CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_DOUT_DEFAULT {0x00000000} CONFIG.C_IS_DUAL {1} \
    CONFIG.C_GPIO2_WIDTH {32} CONFIG.C_ALL_INPUTS_2 {1}] [get_bd_cells capture_control]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_bus
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells control_bus]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_bus
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells memory_bus]

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_GP0] [get_bd_intf_pins control_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_bus/M00_AXI] [get_bd_intf_pins audio_dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins control_bus/M01_AXI] [get_bd_intf_pins capture_control/S_AXI]
connect_bd_intf_net [get_bd_intf_pins capture/M_AXIS] [get_bd_intf_pins stream_fifo/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins stream_fifo/M_AXIS] [get_bd_intf_pins audio_dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins audio_dma/M_AXI_S2MM] [get_bd_intf_pins memory_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_bus/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0]
connect_bd_net [get_bd_pins capture_control/gpio_io_o] [get_bd_pins capture/enable]
connect_bd_net [get_bd_pins capture/status] [get_bd_pins capture_control/gpio2_io_i]

set system_pins {ps/M_AXI_GP0_ACLK ps/S_AXI_HP0_ACLK audio_clock/clk_in1
    reset_system/slowest_sync_clk capture/control_clk capture_control/s_axi_aclk
    audio_dma/s_axi_lite_aclk audio_dma/m_axi_s2mm_aclk stream_fifo/m_axis_aclk
    control_bus/ACLK control_bus/S00_ACLK control_bus/M00_ACLK control_bus/M01_ACLK
    memory_bus/ACLK memory_bus/S00_ACLK memory_bus/M00_ACLK}
foreach pin $system_pins {connect_bd_net [get_bd_pins ps/FCLK_CLK0] [get_bd_pins $pin]}
foreach pin {capture/clk reset_audio/slowest_sync_clk stream_fifo/s_axis_aclk} {
    connect_bd_net [get_bd_pins audio_clock/clk_out1] [get_bd_pins $pin]
}
foreach pin {audio_clock/resetn reset_system/ext_reset_in reset_audio/ext_reset_in} {
    connect_bd_net [get_bd_pins ps/FCLK_RESET0_N] [get_bd_pins $pin]
}
foreach name {reset_system reset_audio} {
    connect_bd_net [get_bd_pins audio_clock/locked] [get_bd_pins $name/dcm_locked]
    connect_bd_net [get_bd_pins zero/dout] [get_bd_pins $name/aux_reset_in] [get_bd_pins $name/mb_debug_sys_rst]
}
foreach pin {audio_dma/axi_resetn capture_control/s_axi_aresetn} {
    connect_bd_net [get_bd_pins reset_system/peripheral_aresetn] [get_bd_pins $pin]
}
foreach pin {control_bus/ARESETN control_bus/S00_ARESETN control_bus/M00_ARESETN control_bus/M01_ARESETN
             memory_bus/ARESETN memory_bus/S00_ARESETN memory_bus/M00_ARESETN} {
    connect_bd_net [get_bd_pins reset_system/interconnect_aresetn] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins reset_audio/peripheral_aresetn] [get_bd_pins capture/resetn] [get_bd_pins stream_fifo/s_axis_aresetn]
create_bd_port -dir I -from 6 -to 0 pdm_data_i
create_bd_port -dir O pdm_clk_o
connect_bd_net [get_bd_ports pdm_data_i] [get_bd_pins capture/pdm_data_i]
connect_bd_net [get_bd_ports pdm_clk_o] [get_bd_pins capture/pdm_clk_o]
assign_bd_address
validate_bd_design
save_bd_design
set design [get_files */capture_system.bd]
generate_target all $design
set wrapper [make_wrapper -files $design -top]
add_files -norecurse $wrapper
set_property top capture_system_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "Created [file join $output horn_capture.xpr]"
if {$mode eq "create"} {return}

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed; inspect the run log"}
open_run synth_1
report_utilization -file [file join $output synthesis_utilization.rpt]
check_timing -verbose -file [file join $output synthesis_check_timing.rpt]
if {$mode eq "synth"} {return}

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed; inspect the run log"}
open_run impl_1
report_timing_summary -report_unconstrained -file [file join $output timing_summary.rpt]
report_cdc -file [file join $output cdc.rpt]
report_drc -file [file join $output drc.rpt]
report_utilization -file [file join $output implementation_utilization.rpt]
check_timing -verbose -file [file join $output implementation_check_timing.rpt]
set paths [get_timing_paths -delay_type min_max -max_paths 10]
if {[llength $paths] == 0} {error "No timing paths reported"}
foreach path $paths {
    if {[get_property SLACK $path] < 0} {error "Timing failed; no bitstream was written"}
}
write_bitstream [file join $output horn_capture.bit]
set handoffs [concat \
    [glob -nocomplain [file join $output horn_capture.srcs sources_1 bd capture_system hw_handoff capture_system.hwh]] \
    [glob -nocomplain [file join $output horn_capture.gen sources_1 bd capture_system hw_handoff capture_system.hwh]]]
if {[llength $handoffs] != 1} {error "Locate capture_system.hwh in the generated block-design output"}
file copy -force [lindex $handoffs 0] [file join $output horn_capture.hwh]
puts "Bitstream written; review timing, CDC, DRC and board measurements before relying on captured audio"
