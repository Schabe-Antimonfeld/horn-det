`timescale 1ns/1ps
module tb_srp_batch;
    reg clk=0, resetn=0, valid=0, last=0, ready=0;
    reg [31:0] data=0;
    reg [3:0] keep=15;
    wire input_ready, output_valid, output_last;
    wire [31:0] output_data;
    wire [3:0] output_keep;
    reg [33:0] packets [0:120000];
    reg [32:0] held;
    reg stalled=0;
    reg [1023:0] input_path, output_path;
    integer words, expected, output_file, received=0, cycle=0, index;
    always #5 clk=~clk;
    srp_batch dut(clk,resetn,data,keep,valid,input_ready,last,
                  output_data,output_keep,output_valid,ready,output_last);
    always @(negedge clk) begin
        cycle=cycle+1;
        ready=(cycle%7 != 0 && cycle%7 != 1);
    end
    always @(posedge clk) if(resetn) begin
        if(stalled && (!output_valid || {output_last,output_data} !== held))
            $fatal(1,"Output changed under backpressure");
        stalled=output_valid && !ready;
        held={output_last,output_data};
        if(output_valid && ready) begin
            if(output_keep !== 15) $fatal(1,"Invalid output keep");
            $fdisplay(output_file,"%09h",{output_last,output_data});
            received=received+1;
        end
    end
    initial begin
        if(!$value$plusargs("INPUT=%s",input_path) || !$value$plusargs("OUTPUT=%s",output_path) ||
           !$value$plusargs("WORDS=%d",words) || !$value$plusargs("EXPECTED=%d",expected))
            $fatal(1,"Missing arguments");
        $readmemh(input_path,packets,0,words-1);
        output_file=$fopen(output_path,"w");
        repeat(3) @(posedge clk);
        @(negedge clk); resetn=1;
        // Reset an incomplete batch before the vectors, checking wrapper/core recovery.
        data=1; valid=1;
        @(posedge clk); @(negedge clk); data=123;
        @(posedge clk); @(negedge clk); resetn=0; valid=0;
        repeat(2) @(posedge clk);
        @(negedge clk); resetn=1;
        for(index=0;index<words;index=index+1) begin
            @(negedge clk);
            data=packets[index][31:0]; last=packets[index][32];
            keep=packets[index][33] ? 4'h7 : 4'hf; valid=1;
            do @(posedge clk); while(!input_ready);
            @(negedge clk); valid=0; last=0;
            if(index%11==0) repeat(2) @(negedge clk);
        end
        wait(received==expected);
        repeat(10) @(posedge clk);
        if(received!=expected) $fatal(1,"Extra output");
        $fclose(output_file);
        $display("PASS BATCH SRP"); $finish;
    end
    initial begin #10000000; $fatal(1,"watchdog"); end
endmodule
