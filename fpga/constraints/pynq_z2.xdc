set_property PACKAGE_PIN W14 [get_ports pdm_clk_o]
set_property IOSTANDARD LVCMOS33 [get_ports pdm_clk_o]
set_property DRIVE 8 [get_ports pdm_clk_o]
set_property SLEW FAST [get_ports pdm_clk_o]
set data_pins {Y14 T11 T10 V16 W16 V12 W13}
for {set index 0} {$index < 7} {incr index} {
    set port [get_ports [format {pdm_data_i[%d]} $index]]
    set_property PACKAGE_PIN [lindex $data_pins $index] $port
    set_property IOSTANDARD LVCMOS33 $port
    set_property PULLUP false $port
    set_property PULLDOWN false $port
}

set source_pin [get_pins -hier -filter {NAME =~ */audio_clock/clk_out1}]
set divided_pin [get_pins -hier -filter {NAME =~ */receiver/pdm_buffer/O}]
if {[llength $source_pin] != 1 || [llength $divided_pin] != 1} {
    error "Cannot identify the audio clock and PDM BUFG; review the synthesized hierarchy"
}
create_generated_clock -name pdm_clock -source $source_pin -divide_by 4 $divided_pin

# Bounds include outbound clock flight and returning data at the FPGA threshold.
if {[info exists ::env(PDM_INPUT_MIN_NS)] && [info exists ::env(PDM_INPUT_MAX_NS)]} {
    set lower $::env(PDM_INPUT_MIN_NS)
    set upper $::env(PDM_INPUT_MAX_NS)
    set ports [get_ports {pdm_data_i[*]}]
    set_input_delay -clock pdm_clock -min $lower $ports
    set_input_delay -clock pdm_clock -max $upper $ports
    set_input_delay -clock pdm_clock -clock_fall -add_delay -min $lower $ports
    set_input_delay -clock pdm_clock -clock_fall -add_delay -max $upper $ports
} else {
    puts "WARNING: PDM input delay bounds are absent; this is not a board-qualified constraint set"
}

# Only first-stage control/status synchronizers are exempted, never PDM inputs.
set synchronizers [get_cells -hier -filter {NAME =~ */enable_meta_reg || NAME =~ */status_meta_reg*}]
if {[llength $synchronizers] > 0} {
    set_false_path -to [get_pins -of_objects $synchronizers -filter {REF_PIN_NAME == D}]
}
