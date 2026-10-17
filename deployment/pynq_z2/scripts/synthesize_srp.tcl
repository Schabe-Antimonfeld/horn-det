if {[llength $argv] != 1} {error "Usage: synthesize_srp.tcl <new_output_directory>"}
set srp_output [file normalize [lindex $argv 0]]
if {[file exists $srp_output]} {error "Use a new output directory"}
file mkdir $srp_output
set srp_source [file normalize [file join [file dirname [info script]] .. rtl srp_accumulator.v]]
create_project -in_memory -part xc7z020clg400-1
read_verilog $srp_source
synth_design -top srp_accumulator -part xc7z020clg400-1
create_clock -period 10 [get_ports clk]
report_utilization -file [file join $srp_output utilization.rpt]
report_timing_summary -file [file join $srp_output timing.rpt]
write_checkpoint [file join $srp_output srp.dcp]
