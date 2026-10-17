if {[llength $argv] < 3 || [llength $argv] > 4} {
    error "Usage: build.tcl <model_directory> <new_build_directory> <board_repository_or_dash> ?create|synth|bit?"
}
set deploy_root [file normalize [file join [file dirname [info script]] ..]]
set deploy_vendor [file join $deploy_root vendor fpga]
if {![file exists [file join $deploy_vendor scripts build_cnn.tcl]]} {
    set deploy_vendor [file normalize [file join $deploy_root .. .. fpga]]
}
if {![file exists [file join $deploy_vendor scripts build_cnn.tcl]]} {
    error "FPGA sources missing; use a complete deployment package or source checkout"
}
set cnn_extension [file join $deploy_root scripts connect_srp.tcl]
source [file join $deploy_vendor scripts build_cnn.tcl]
