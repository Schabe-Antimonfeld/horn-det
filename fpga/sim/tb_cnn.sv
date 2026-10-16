`timescale 1ns / 1ps

module tb_cnn;
    parameter MODEL_DIR = ".";
    parameter FRAMES = 4;
    reg clk = 0;
    always #5 clk = !clk;
    reg resetn = 0;
    reg [31:0] input_data = 0;
    reg [3:0] input_keep = 4'hf;
    reg input_valid = 0;
    reg input_last = 0;
    wire input_ready;
    wire [31:0] output_data;
    wire [3:0] output_keep;
    wire output_valid, output_last;
    reg output_ready = 0;
    reg [31:0] inputs [0:FRAMES*960-1];
    reg [31:0] expected [0:FRAMES*8-1];
    integer seed = 1729;
    integer packets = 0;
    integer word_index = 0;
    integer golden_index;
    reg stalled = 0;
    reg [31:0] previous_data;
    reg previous_last;
    reg error_packet;

    tiny_cnn #(.MODEL_DIR(MODEL_DIR)) dut (
        .clk(clk), .resetn(resetn),
        .s_axis_tdata(input_data), .s_axis_tkeep(input_keep),
        .s_axis_tvalid(input_valid), .s_axis_tready(input_ready), .s_axis_tlast(input_last),
        .m_axis_tdata(output_data), .m_axis_tkeep(output_keep),
        .m_axis_tvalid(output_valid), .m_axis_tready(output_ready), .m_axis_tlast(output_last)
    );

    initial begin
        $readmemh({MODEL_DIR, "/inputs.mem"}, inputs);
        $readmemh({MODEL_DIR, "/expected.mem"}, expected);
    end

    task automatic send_word(input [31:0] value, input last, input [3:0] keep);
        begin
            repeat (($random(seed) & 3) == 0 ? 1 : 0) @(negedge clk);
            @(negedge clk);
            input_data = value;
            input_valid = 1;
            input_last = last;
            input_keep = keep;
            @(posedge clk);
            while (!input_ready) @(posedge clk);
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    task automatic send_frame(input integer index, input finish);
        integer position;
        begin
            for (position = 0; position < 960; position = position + 1)
                send_word(inputs[index*960+position], finish && position == 959, 4'hf);
        end
    endtask

    task automatic send_corrupted(input integer mode);
        integer position;
        reg [31:0] value;
        begin
            for (position = 0; position < 960; position = position+1) begin
                value = inputs[position];
                if (mode == 1 && position == 13) value = value ^ 32'hffff0000;
                send_word(value, position == 959, mode == 0 && position == 13 ? 4'h7 : 4'hf);
            end
        end
    endtask

    always @(negedge clk)
        output_ready = ($random(seed) & 3) != 0;

    always @(posedge clk) begin
        if (!resetn) begin
            packets = 0; word_index = 0; stalled = 0;
        end else begin
            if (stalled && (!output_valid || output_data !== previous_data || output_last !== previous_last))
                $fatal(1, "CNN output changed under backpressure");
            stalled = output_valid && !output_ready;
            previous_data = output_data;
            previous_last = output_last;
            if (output_valid && input_ready) $fatal(1, "CNN accepted another frame during output");
            if (output_valid && output_ready) begin
                if (output_keep !== 4'hf || output_last !== (word_index == 7))
                    $fatal(1, "CNN output framing mismatch");
                error_packet = packets == FRAMES || packets == FRAMES+2 || packets == FRAMES+3 || packets == FRAMES+4;
                golden_index = packets < FRAMES ? packets : 0;
                if (word_index == 2) begin
                    if (output_data !== packets) $fatal(1, "CNN inference sequence mismatch");
                end else if (word_index == 6) begin
                    if (error_packet && output_data !== 0) $fatal(1, "Invalid frame executed inference");
                    if (!error_packet && (output_data == 0 || output_data > 1066666))
                        $fatal(1, "CNN service-cycle budget exceeded: %0d", output_data);
                    if (!error_packet) $display("CNN frame %0d compute cycles=%0d", packets, output_data);
                end else if (error_packet && word_index >= 3 && word_index <= 5) begin
                    if (output_data !== (word_index == 5 ? 32'h00020000 : 0))
                        $fatal(1, "Invalid input did not produce an explicit error result");
                end else if (output_data !== expected[golden_index*8+word_index]) begin
                    $fatal(1, "CNN mismatch packet=%0d word=%0d got=%h expected=%h layer=%0d", packets,
                           word_index, output_data, expected[golden_index*8+word_index], dut.layer);
                end
                if (output_last) begin packets = packets+1; word_index = 0; end
                else word_index = word_index+1;
            end
        end
    end

    integer frame;
    initial begin
        repeat (5) @(negedge clk);
        if (dut.requantize(48'sd128, 6'd8, 0) !== 17'h00001 ||
            dut.requantize(-48'sd128, 6'd8, 0) !== 17'h0ffff ||
            dut.requantize(-48'sd128, 6'd8, 1) !== 17'h00000 ||
            dut.requantize(48'sd8388608, 6'd8, 0) !== 17'h17fff ||
            dut.requantize(-48'sd8388608, 6'd8, 0) !== 17'h08000)
            $fatal(1, "CNN requantization boundary mismatch");
        if (dut.probability_address(-16'sd32768, 16'sd32767, 5'd15) !== 11'd640 ||
            dut.probability_address(16'sd32767, -16'sd32768, 5'd15) !== 11'd384 ||
            dut.probability_address(-16'sd32768, 16'sd32767, 5'd0) !== 11'd1024)
            $fatal(1, "CNN signed softmax index mismatch");
        resetn = 1;
        for (frame = 0; frame < FRAMES; frame = frame+1) send_frame(frame, 1);
        wait (packets == FRAMES);
        send_word(32'd0, 0, 4'hf);
        send_word(32'd0, 1, 4'h7);
        wait (packets == FRAMES+1);
        send_frame(0, 1);
        wait (packets == FRAMES+2);
        send_frame(0, 0);
        send_word(32'd0, 1, 4'hf);
        wait (packets == FRAMES+3);
        send_corrupted(0);
        wait (packets == FRAMES+4);
        send_corrupted(1);
        wait (packets == FRAMES+5);
        send_frame(0, 1);
        wait (dut.state == 3);
        @(negedge clk); resetn = 0;
        repeat (5) @(negedge clk);
        resetn = 1;
        send_frame(0, 1);
        wait (packets == 1);
        $display("PASS CNN integer reference, padding, depthwise/pointwise, GAP, softmax, framing, stalls and reset");
        $finish;
    end

    initial begin
        #60000000;
        $fatal(1, "CNN simulation timeout");
    end
endmodule
