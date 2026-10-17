`timescale 1ns/1ps
module tb_srp;
    reg clk=0, resetn=0, valid=0, last=0, ready=0;
    reg [31:0] data=0;
    wire input_ready, output_valid, output_last;
    wire [31:0] output_data;
    reg [31:0] received [0:3];
    integer index;
    reg signed [63:0] actual;
    always #5 clk=~clk;
    srp_accumulator #(.PAIRS(2)) dut(clk,resetn,data,4'hf,valid,input_ready,last,
                                    output_data,,output_valid,ready,output_last);
    task send(input [31:0] word, input ending);
        begin
            @(negedge clk); data=word; valid=1; last=ending;
            do @(posedge clk); while (!input_ready);
            @(negedge clk); valid=0; last=0;
        end
    endtask
    task receive;
        begin
            wait(output_valid);
            repeat(5) begin @(posedge clk); if(output_data!==32'h31505253) $fatal(1,"unstable header"); end
            @(negedge clk); ready=1;
            for(index=0;index<4;index=index+1) begin
                @(posedge clk);
                if(!output_valid || output_last!==(index==3)) $fatal(1,"output protocol");
                received[index]=output_data;
            end
            @(negedge clk); ready=0;
        end
    endtask
    initial begin
        repeat(3) @(posedge clk); @(negedge clk); resetn=1;
        send(-100,0); send(300,0); send(16384,0);
        send(1000,0); send(-1000,0); send(32768,1);
        receive(); actual={received[2],received[1]};
        if(actual!==0 || received[3]!==0) $fatal(1,"signed interpolation mismatch");
        send(700,1); receive();
        if(received[3]!==1) $fatal(1,"early TLAST not rejected");
        send(-8388608,0); send(8388607,0); send(0,0);
        send(-1,0); send(-1,0); send(65535,1);
        receive(); actual={received[2],received[1]};
        if(actual!==-64'sd549755879424 || received[3]!==0) $fatal(1,"extreme mismatch %d",actual);
        send(123,0); @(negedge clk); resetn=0; repeat(2) @(posedge clk);
        @(negedge clk); resetn=1;
        send(10,0); send(10,0); send(0,0); send(20,0); send(20,0); send(0,1);
        receive(); if(received[1]!==1966080 || received[3]!==0) $fatal(1,"reset recovery");
        $display("PASS SRP interpolation, signed extremes, stalls, packet errors and reset"); $finish;
    end
    initial begin #100000; $fatal(1,"watchdog"); end
endmodule
