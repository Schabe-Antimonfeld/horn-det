if {[llength $argv] < 2 || [llength $argv] > 3} {
    error "Usage: build_ps.tcl <new_build_directory> <board_repository_or_dash> ?create|synth|bit?"
}
set capture_root [file normalize [file join [file dirname [info script]] ..]]
set capture_vendor [file join $capture_root vendor fpga]
if {![file exists [file join $capture_vendor scripts build.tcl]]} {
    set capture_vendor [file normalize [file join $capture_root .. .. fpga]]
}
if {![file exists [file join $capture_vendor scripts build.tcl]]} {
    error "Capture sources missing; use a complete deployment package or source checkout"
}
source [file join $capture_vendor scripts build.tcl]
