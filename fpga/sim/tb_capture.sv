`timescale 1ns / 1ps

module tb_capture;
    parameter PAIRS = 7;
    localparam CHANNELS = 2*PAIRS;
    reg clk = 0;
    reg control_clk = 0;
    always #40.690104 clk = !clk;
    always #5 control_clk = !control_clk;
    reg resetn = 0;
    reg enable = 0;
    tri [PAIRS-1:0] data_bus;
    wire pdm_clock;
    wire [31:0] top_data, status;
    wire [3:0] top_keep;
    wire top_valid, top_last;
    reg top_ready = 0;
    integer serial_index = 0;
    integer seed = 71231;
    integer filter_done = 0;
    integer packets_done = 0;

    pdm_capture #(.PAIRS(PAIRS), .HOP(8), .WARMUP_BITS(128)) dut (
        .clk(clk), .control_clk(control_clk), .resetn(resetn), .enable(enable),
        .pdm_data_i(data_bus), .pdm_clk_o(pdm_clock),
        .m_axis_tdata(top_data), .m_axis_tkeep(top_keep),
        .m_axis_tvalid(top_valid), .m_axis_tready(top_ready),
        .m_axis_tlast(top_last), .status(status)
    );

    function automatic pattern(input integer serial, input integer channel);
        reg [31:0] mixed;
        begin
            mixed = serial*32'd1103515245 + channel*32'd12345;
            if (channel == 0) pattern = 1;
            else if (channel == 1) pattern = 0;
            else if (channel == 2) pattern = serial & 1;
            else pattern = mixed[(channel % 11)+1];
        end
    endfunction

    genvar pair_index;
    generate for (pair_index = 0; pair_index < PAIRS; pair_index = pair_index + 1) begin: microphones
        // Two independent tri-state drivers model delayed bus handover.
        assign #(50, 50, 10) data_bus[pair_index] = pdm_clock ? 1'bz : pattern(serial_index, 2*pair_index);
        assign #(50, 50, 10) data_bus[pair_index] = pdm_clock ? pattern(serial_index, 2*pair_index+1) : 1'bz;
    end endgenerate

    always @(negedge pdm_clock or negedge resetn)
        if (!resetn) serial_index = 0;
        else serial_index = serial_index + 1;

    reg [CHANNELS-1:0] expected_rx [0:65535];
    integer rx_write = 0;
    integer rx_read = 0;
    integer source_channel;
    always @(posedge pdm_clock) begin
        if (resetn) begin
            for (source_channel = 0; source_channel < CHANNELS; source_channel = source_channel + 1)
                expected_rx[rx_write][source_channel] = pattern(serial_index, source_channel);
            rx_write = rx_write + 1;
        end
    end

    integer kernel [0:155];
    integer temporary [0:155];
    integer length, stage, i, j;
    integer coefficients [0:62];
    initial begin
        for (i = 0; i < 156; i = i + 1) kernel[i] = 0;
        kernel[0] = 1;
        length = 1;
        // An FIR convolution of five boxcars is independent of the RTL integrators.
        for (stage = 0; stage < 5; stage = stage + 1) begin
            for (i = 0; i < 156; i = i + 1) temporary[i] = 0;
            for (i = 0; i < length; i = i + 1)
                for (j = 0; j < 32; j = j + 1)
                    temporary[i+j] = temporary[i+j] + kernel[i];
            length = length + 31;
            for (i = 0; i < 156; i = i + 1) kernel[i] = temporary[i];
        end
        coefficients[0]=-1; coefficients[1]=-9; coefficients[2]=-21;
        coefficients[3]=-12; coefficients[4]=37; coefficients[5]=73;
        coefficients[6]=1; coefficients[7]=-146; coefficients[8]=-142;
        coefficients[9]=139; coefficients[10]=372; coefficients[11]=79;
        coefficients[12]=-552; coefficients[13]=-573; coefficients[14]=420;
        coefficients[15]=1213; coefficients[16]=288; coefficients[17]=-1599;
        coefficients[18]=-1618; coefficients[19]=1141; coefficients[20]=3188;
        coefficients[21]=688; coefficients[22]=-4094; coefficients[23]=-4008;
        coefficients[24]=3003; coefficients[25]=8212; coefficients[26]=1774;
        coefficients[27]=-11656; coefficients[28]=-13156; coefficients[29]=9451;
        coefficients[30]=43270; coefficients[31]=59548;
        for (i = 32; i < 63; i = i + 1) coefficients[i] = coefficients[62-i];
    end

    function automatic longint signed rounded(input longint signed value, input integer shift);
        if (value >= 0) rounded = (value + (64'sd1 << (shift-1))) >>> shift;
        else rounded = -(((-value) + (64'sd1 << (shift-1))) >>> shift);
    endfunction

    integer pdm_history [0:CHANNELS-1][0:155];
    integer fir_history [0:CHANNELS-1][0:62];
    reg signed [15:0] expected_filter [0:2047][0:CHANNELS-1];
    reg signed [15:0] usable [0:2047][0:CHANNELS-1];
    integer pdm_seen = 0;
    integer cic_seen = 0;
    integer history_index = 0;
    integer filter_write = 0;
    integer filter_read = 0;
    integer usable_count = 0;
    integer channel, k;
    longint signed total, scaled;
    always @(negedge clk) begin
        if (!resetn) begin
            rx_write = 0; rx_read = 0;
            pdm_seen = 0; cic_seen = 0; history_index = 0;
            filter_write = 0; filter_read = 0; usable_count = 0;
            for (channel = 0; channel < CHANNELS; channel = channel + 1)
                for (k = 0; k < 63; k = k + 1) fir_history[channel][k] = 0;
        end else begin
            if (dut.pdm_valid) begin
                if (rx_read >= rx_write || dut.pdm_samples !== expected_rx[rx_read])
                    $fatal(1, "PDM edge/channel mismatch at bit %0d: %h expected %h",
                           rx_read, dut.pdm_samples, expected_rx[rx_read]);
                for (channel = 0; channel < CHANNELS; channel = channel + 1) begin
                    pdm_history[channel][history_index] = expected_rx[rx_read][channel] ? 1 : -1;
                    if (pdm_seen % 32 == 31) begin
                        total = 0;
                        for (k = 0; k < 156; k = k + 1)
                            if (pdm_seen >= k)
                                total = total + longint'(kernel[k])*pdm_history[channel][(history_index-k+156)%156];
                        scaled = rounded(total, 2);
                        if (scaled > 8388607) scaled = 8388607;
                        if (scaled < -8388608) scaled = -8388608;
                        for (k = 62; k > 0; k = k - 1)
                            fir_history[channel][k] = fir_history[channel][k-1];
                        fir_history[channel][0] = scaled;
                        if (cic_seen % 2 == 1) begin
                            total = 0;
                            for (k = 0; k < 63; k = k + 1)
                                total = total + longint'(coefficients[k])*fir_history[channel][k];
                            scaled = rounded(total, 25);
                            if (scaled > 32767) scaled = 32767;
                            if (scaled < -32768) scaled = -32768;
                            expected_filter[filter_write][channel] = scaled;
                        end
                    end
                end
                if (pdm_seen % 32 == 31) begin
                    if (cic_seen % 2 == 1) filter_write = filter_write + 1;
                    cic_seen = cic_seen + 1;
                end
                pdm_seen = pdm_seen + 1;
                history_index = (history_index+1)%156;
                rx_read = rx_read + 1;
            end
            if (dut.pcm_valid) begin
                if (filter_read >= filter_write) $fatal(1, "Unexpected decimator output");
                for (channel = 0; channel < CHANNELS; channel = channel + 1) begin
                    if ($signed(dut.pcm_samples[16*channel +: 16]) !== expected_filter[filter_read][channel])
                        $fatal(1, "Filter mismatch frame=%0d ch=%0d got=%0d expected=%0d", filter_read, channel,
                               $signed(dut.pcm_samples[16*channel +: 16]), expected_filter[filter_read][channel]);
                    if (dut.ready)
                        usable[usable_count][channel] = expected_filter[filter_read][channel];
                end
                filter_read = filter_read + 1;
                if (dut.ready) usable_count = usable_count + 1;
            end
            top_ready = ($random(seed) & 3) != 0;
        end
    end

    integer top_word = 0;
    integer top_first = 0;
    integer top_packets = 0;
    reg top_stalled = 0;
    reg [31:0] top_previous;
    reg top_previous_last;
    always @(posedge clk) begin
        if (!resetn) begin
            top_word = 0; top_first = 0; top_packets = 0; top_stalled = 0;
        end else begin
            if (top_stalled && (!top_valid || top_data !== top_previous || top_last !== top_previous_last))
                $fatal(1, "Integrated stream changed under backpressure");
            top_stalled = top_valid && !top_ready;
            top_previous = top_data;
            top_previous_last = top_last;
            if (top_valid && top_ready) begin
                if (top_keep !== 4'hf || top_last !== (top_word == 8+8*CHANNELS-1))
                    $fatal(1, "Integrated packet framing mismatch");
                case (top_word)
                    0: if (top_data !== 32'h314d4450) $fatal(1, "Bad magic");
                    1: if (top_data !== (32'h00010800 | CHANNELS)) $fatal(1, "Bad format");
                    2: top_first = top_data;
                    3: if (top_data !== 0) $fatal(1, "Bad high sample counter");
                    4: if (top_data !== 0) $fatal(1, "Unexpected integrated drop");
                    5: if (top_data !== 8) $fatal(1, "Bad hop");
                    6: if (top_data !== 48000) $fatal(1, "Bad sample rate");
                    7: if (top_data[31:1] !== 0) $fatal(1, "Reserved flag set");
                    default: begin
                        if (top_data !== {{16{usable[top_first+(top_word-8)/CHANNELS][(top_word-8)%CHANNELS][15]}},
                                         usable[top_first+(top_word-8)/CHANNELS][(top_word-8)%CHANNELS]})
                            $fatal(1, "Integrated payload mismatch word=%0d", top_word);
                    end
                endcase
                if (top_last) begin top_word = 0; top_packets = top_packets + 1; end
                else top_word = top_word + 1;
            end
        end
    end

    reg packet_resetn = 0;
    reg packet_enable = 1;
    reg [16*CHANNELS-1:0] packet_samples = 0;
    reg packet_valid = 0;
    wire [31:0] packet_data, drops;
    wire [3:0] packet_keep;
    wire packet_output, packet_last;
    reg packet_ready = 0;
    axis_pcm #(.CHANNELS(CHANNELS)) packet_dut (
        .clk(clk), .resetn(packet_resetn), .enable(packet_enable),
        .samples(packet_samples), .valid(packet_valid), .clipped(1'b0),
        .m_axis_tdata(packet_data), .m_axis_tkeep(packet_keep),
        .m_axis_tvalid(packet_output), .m_axis_tready(packet_ready),
        .m_axis_tlast(packet_last), .dropped(drops)
    );

    function automatic [31:0] sample_word(input integer frame, input integer channel);
        reg signed [15:0] value;
        begin
            value = ((frame*73 + channel*911)%65536)-32768;
            sample_word = {{16{value[15]}}, value};
        end
    endfunction

    task automatic push_block(input integer block_index);
        integer frame, ch;
        begin
            for (frame = 0; frame < 512; frame = frame + 1) begin
                @(negedge clk);
                packet_valid = 1;
                for (ch = 0; ch < CHANNELS; ch = ch + 1)
                    packet_samples[16*ch +: 16] = sample_word(block_index*512+frame, ch);
                @(negedge clk);
                packet_valid = 0;
                repeat (18) @(negedge clk);
            end
        end
    endtask

    integer packet_word = 0;
    integer packet_first = 0;
    integer packet_count = 0;
    reg packet_stalled = 0;
    reg [31:0] previous_data;
    reg previous_last;
    integer expected_first;
    always @(posedge clk) begin
        if (!packet_resetn) begin
            packet_word = 0; packet_count = 0; packet_first = 0; packet_stalled = 0;
        end else begin
            if (packet_stalled && (!packet_output || packet_data !== previous_data || packet_last !== previous_last))
                $fatal(1, "Packet stream changed under backpressure");
            packet_stalled = packet_output && !packet_ready;
            previous_data = packet_data;
            previous_last = packet_last;
            if (packet_output && packet_ready) begin
                if (packet_keep !== 4'hf || packet_last !== (packet_word == 8+512*CHANNELS-1))
                    $fatal(1, "TLAST/TKEEP mismatch");
                case (packet_count)
                    0: expected_first = 0;
                    1: expected_first = 512;
                    2: expected_first = 1536;
                    3: expected_first = 2560;
                    default: expected_first = 3072;
                endcase
                case (packet_word)
                    0: if (packet_data !== 32'h314d4450) $fatal(1, "Packet magic mismatch");
                    1: if (packet_data !== (32'h00010800 | CHANNELS)) $fatal(1, "Packet format mismatch");
                    2: begin
                        packet_first = packet_data;
                        if (packet_first != expected_first) $fatal(1, "Sample discontinuity missing");
                    end
                    3: if (packet_data !== 0) $fatal(1, "High counter mismatch");
                    4: if (packet_data !== (packet_count >= 2 ? 32'd1 : 32'd0)) $fatal(1, "Loss count mismatch");
                    5: if (packet_data !== 512) $fatal(1, "Hop mismatch");
                    6: if (packet_data !== 48000) $fatal(1, "Rate mismatch");
                    7: if (packet_data !== 0) $fatal(1, "Flags mismatch");
                    default: if (packet_data !== sample_word(packet_first+(packet_word-8)/CHANNELS, (packet_word-8)%CHANNELS))
                        $fatal(1, "Packet channel/sample order mismatch at %0d", packet_word);
                endcase
                if (packet_last) begin packet_word = 0; packet_count = packet_count + 1; end
                else packet_word = packet_word + 1;
            end
        end
    end

    initial begin
        repeat (8) @(negedge clk);
        if (dut.decimator.cic_scale(32'sd2) !== 24'sd1 ||
            dut.decimator.cic_scale(-32'sd2) !== -24'sd1)
            $fatal(1, "CIC half-way rounding mismatch");
        if (dut.decimator.pcm_scale(48'sd16777216) !== 17'h00001 ||
            dut.decimator.pcm_scale(-48'sd16777216) !== 17'h0ffff ||
            dut.decimator.pcm_scale(48'sd16777215) !== 17'h00000 ||
            dut.decimator.pcm_scale(-48'sd16777215) !== 17'h00000 ||
            dut.decimator.pcm_scale(48'sd1099511627776) !== 17'h17fff ||
            dut.decimator.pcm_scale(-48'sd1099511627776) !== 17'h08000 ||
            dut.decimator.pcm_scale(-48'sd1099545182208) !== 17'h18000)
            $fatal(1, "PCM rounding/saturation boundary mismatch");
        resetn = 1;
        wait (status[1]);
        @(negedge clk); enable = 1;
        wait (filter_read >= 96 && top_packets >= 5);
        if (expected_filter[95][0] !== 16'h7fff || expected_filter[95][1] !== 16'h8000)
            $fatal(1, "DC gain/saturation endpoint mismatch");
        @(negedge clk); resetn = 0; enable = 0;
        repeat (8) @(negedge clk);
        resetn = 1;
        wait (status[1]);
        @(negedge clk); enable = 1;
        wait (filter_read >= 64 && top_packets >= 3);
        filter_done = 1;
        $display("PASS: delayed dual-edge PDM, independent CIC/FIR reference, integrated packets, warmup and reset");
    end

    initial begin
        repeat (8) @(negedge clk);
        packet_resetn = 1;
        push_block(0);
        push_block(1);
        push_block(2);
        if (drops !== 1) $fatal(1, "Expected one whole-block drop");
        @(negedge clk); packet_ready = 1;
        wait (packet_count == 2);
        push_block(3);
        wait (packet_count == 3);
        @(negedge clk); packet_enable = 0;
        push_block(4);
        @(negedge clk); packet_enable = 1;
        push_block(5);
        wait (packet_count == 4);
        if (drops !== 1) $fatal(1, "Disabled acquisition counted as loss");
        push_block(6);
        wait (packet_word > 100);
        @(negedge clk); packet_ready = 0; packet_resetn = 0;
        repeat (8) @(negedge clk);
        packet_resetn = 1; packet_ready = 1;
        push_block(0);
        wait (packet_count == 1);
        if (drops !== 0) $fatal(1, "Reset did not clear drops");
        packets_done = 1;
        $display("PASS: 512-frame DMA packets, signed payloads, stalls, whole-block overflow, disable and reset");
    end

    initial begin
        wait (filter_done && packets_done);
        $display("PASS: all capture tests, CHANNELS=%0d", CHANNELS);
        $finish;
    end
    initial begin
        #12000000;
        $fatal(1, "Simulation timeout");
    end
endmodule
