if {[llength $argv] < 3 || [llength $argv] > 4} {
    error "Usage: build_cnn.tcl <model_directory> <new_build_directory> <board_repository_or_dash> ?create|synth|bit?"
}
set cnn_model [file normalize [lindex $argv 0]]
set cnn_output [file normalize [lindex $argv 1]]
set cnn_board [lindex $argv 2]
set cnn_mode [expr {[llength $argv] == 4 ? [lindex $argv 3] : "create"}]
set cnn_root [file normalize [file join [file dirname [info script]] ..]]
if {$cnn_mode ni {create synth bit}} {error "Mode must be create, synth or bit"}
if {[file exists [file join $cnn_output horn_capture.xpr]]} {error "Choose a new build directory"}
if {$cnn_mode eq "bit"} {
    foreach name {PDM_INPUT_MIN_NS PDM_INPUT_MAX_NS} {
        if {![info exists ::env($name)] || ![string is double -strict $::env($name)]} {
            error "Set $name from the verified PDM interface budget before implementation"
        }
    }
    if {$::env(PDM_INPUT_MIN_NS) < 0 || $::env(PDM_INPUT_MAX_NS) <= $::env(PDM_INPUT_MIN_NS) ||
        $::env(PDM_INPUT_MAX_NS) >= 156.25} {error "Invalid PDM interface delay bounds"}
}
set cnn_python [expr {[info exists ::env(CNN_PYTHON)] ? $::env(CNN_PYTHON) : "python"}]
if {[catch {exec $cnn_python -E [file join $cnn_root cnn.py] check --model-dir $cnn_model} message]} {
    error "Model validation failed; set CNN_PYTHON to the prototype interpreter if necessary: $message"
}
set saved_arguments $argv
set argv [list $cnn_output $cnn_board create]
source [file join $cnn_root scripts build.tcl]
set argv $saved_arguments

set cnn_snapshot [file join $cnn_output model]
file mkdir $cnn_snapshot
foreach path [concat [glob [file join $cnn_model *.mem]] [list [file join $cnn_model manifest.json]]] {
    file copy $path [file join $cnn_snapshot [file tail $path]]
}
add_files -norecurse [file join $cnn_root rtl tiny_cnn.v]
add_files -norecurse [glob [file join $cnn_snapshot *.mem]]
update_compile_order -fileset sources_1
open_bd_design [get_files */capture_system.bd]
create_bd_cell -type module -reference tiny_cnn inference
set_property CONFIG.MODEL_DIR $cnn_snapshot [get_bd_cells inference]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 cnn_dma
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16}] [get_bd_cells cnn_dma]
set_property CONFIG.NUM_MI {3} [get_bd_cells control_bus]
set_property CONFIG.NUM_SI {3} [get_bd_cells memory_bus]
connect_bd_intf_net [get_bd_intf_pins control_bus/M02_AXI] [get_bd_intf_pins cnn_dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins cnn_dma/M_AXI_MM2S] [get_bd_intf_pins memory_bus/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins cnn_dma/M_AXI_S2MM] [get_bd_intf_pins memory_bus/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins cnn_dma/M_AXIS_MM2S] [get_bd_intf_pins inference/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins inference/M_AXIS] [get_bd_intf_pins cnn_dma/S_AXIS_S2MM]
foreach pin {inference/clk cnn_dma/s_axi_lite_aclk cnn_dma/m_axi_mm2s_aclk cnn_dma/m_axi_s2mm_aclk
             control_bus/M02_ACLK memory_bus/S01_ACLK memory_bus/S02_ACLK} {
    connect_bd_net [get_bd_pins ps/FCLK_CLK0] [get_bd_pins $pin]
}
foreach pin {inference/resetn cnn_dma/axi_resetn} {
    connect_bd_net [get_bd_pins reset_system/peripheral_aresetn] [get_bd_pins $pin]
}
foreach pin {control_bus/M02_ARESETN memory_bus/S01_ARESETN memory_bus/S02_ARESETN} {
    connect_bd_net [get_bd_pins reset_system/interconnect_aresetn] [get_bd_pins $pin]
}
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files */capture_system.bd]
update_compile_order -fileset sources_1
puts "Created capture+CNN project; audio_dma and cnn_dma share PS HP0"
if {$cnn_mode eq "create"} {return}

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "CNN synthesis failed"}
open_run synth_1
report_utilization -file [file join $cnn_output cnn_synthesis_utilization.rpt]
check_timing -verbose -file [file join $cnn_output cnn_synthesis_timing.rpt]
if {$cnn_mode eq "synth"} {return}
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "CNN implementation failed"}
open_run impl_1
report_utilization -file [file join $cnn_output cnn_implementation_utilization.rpt]
report_timing_summary -report_unconstrained -file [file join $cnn_output cnn_timing_summary.rpt]
report_cdc -file [file join $cnn_output cnn_cdc.rpt]
report_drc -file [file join $cnn_output cnn_drc.rpt]
check_timing -verbose -file [file join $cnn_output cnn_check_timing.rpt]
set cnn_paths [get_timing_paths -delay_type min_max -max_paths 10]
if {[llength $cnn_paths] == 0} {error "No CNN timing paths reported"}
foreach path $cnn_paths {
    if {[get_property SLACK $path] < 0} {error "Timing failed; no CNN bitstream written"}
}
write_bitstream [file join $cnn_output horn_cnn.bit]
set cnn_handoffs [concat \
    [glob -nocomplain [file join $cnn_output horn_capture.srcs sources_1 bd capture_system hw_handoff capture_system.hwh]] \
    [glob -nocomplain [file join $cnn_output horn_capture.gen sources_1 bd capture_system hw_handoff capture_system.hwh]]]
if {[llength $cnn_handoffs] != 1} {error "Locate the generated capture_system.hwh"}
file copy [lindex $cnn_handoffs 0] [file join $cnn_output horn_cnn.hwh]
puts "CNN bitstream uses the model snapshot in $cnn_snapshot; review all reports before board use"
