puts "DEPLOY_VERSION:[version -short]"
create_project -in_memory -part xc7z020clg400-1
puts "DEPLOY_PART:[get_parts -quiet xc7z020clg400-1]"
puts "DEPLOY_BOARDS:[get_board_parts -quiet -filter {NAME =~ *pynq-z2*}]"
foreach name {processing_system7 axi_dma xfft} {
    puts "DEPLOY_IP_${name}:[get_ipdefs -quiet xilinx.com:ip:${name}:*]"
}
exit
