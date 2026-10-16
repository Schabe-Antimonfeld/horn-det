`timescale 1ns / 1ps

module tiny_cnn #(
    parameter MODEL_DIR = "."
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK",
       X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET resetn, FREQ_HZ 100000000" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire resetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input wire [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output reg [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast
);
    localparam LOAD = 0, SETUP = 1, READ_MAC = 2, MAC = 3, QUANT = 4, STORE = 5,
               POOL_INIT = 6, POOL_READ = 7, POOL_ADD = 8, PROB = 9, OUTPUT = 10, DRAIN = 11;
    reg [3:0] state;
    reg [2:0] layer;
    reg source_bank;
    reg [1:0] group_index;
    reg [9:0] position;
    reg [5:0] row;
    reg [4:0] column;
    reg [5:0] term;
    reg [9:0] input_index;
    reg input_bad;
    reg [2:0] output_index;
    reg [31:0] frame_id, next_id, cycles;
    reg [15:0] flags;
    reg signed [15:0] score0, score1;
    reg [15:0] probability;
    reg [31:0] configuration [0:8];
    reg [15:0] sigmoid [0:1024];
    reg signed [15:0] pooled [0:31];
    wire [15:0] read_value [0:7];
    wire [15:0] stored_value [0:7];
    wire signed [31:0] pool_value [0:7];
    wire [7:0] quant_clips;
    reg [11:0] read_address;
    reg input_inside;
    reg sampled_inside;
    reg [2:0] input_lane;
    reg [2:0] sampled_lane;
    integer neighbor_row, neighbor_column;
    wire depthwise = layer == 1 || layer == 3;
    wire kernel = layer == 0 || depthwise;
    wire [5:0] terms = kernel ? 6'd9 : layer == 2 ? 6'd16 : layer == 4 ? 6'd24 : 6'd32;
    wire [1:0] final_group = layer == 0 || layer == 1 ? 2'd1 :
                            layer == 2 || layer == 3 ? 2'd2 : layer == 4 ? 2'd3 : 2'd0;
    wire [7:0] weight_base = layer == 0 ? 0 : layer == 1 ? 18 : layer == 2 ? 36 :
                             layer == 3 ? 84 : layer == 4 ? 111 : 207;
    wire [3:0] bias_base = layer == 0 ? 0 : layer == 1 ? 2 : layer == 2 ? 4 :
                           layer == 3 ? 7 : layer == 4 ? 10 : 14;
    wire [7:0] weight_address = weight_base + group_index*terms + term;
    wire [11:0] write_address = group_index*960 + position;
    wire read_enable = state == READ_MAC || state == POOL_READ;
    wire load_enable = state == LOAD && s_axis_tvalid;
    wire bad_word = s_axis_tkeep != 4'hf || s_axis_tdata[31:16] != {16{s_axis_tdata[15]}};
    wire [7:0] active_mask = layer == 5 ? 8'h03 : 8'hff;

    initial begin
        $readmemh({MODEL_DIR, "/config.mem"}, configuration);
        $readmemh({MODEL_DIR, "/softmax.mem"}, sigmoid);
    end

    assign s_axis_tready = state == LOAD || state == DRAIN;
    assign m_axis_tvalid = state == OUTPUT;
    assign m_axis_tkeep = 4'hf;
    assign m_axis_tlast = state == OUTPUT && output_index == 7;

    function [16:0] requantize;
        input signed [47:0] value;
        input [5:0] shift;
        input relu;
        reg signed [48:0] wide, rounded, half;
        begin
            wide = value;
            half = shift == 0 ? 49'sd0 : 49'sd1 << (shift-1);
            if (relu && wide < 0)
                rounded = 0;
            else if (wide >= 0)
                rounded = (wide + half) >>> shift;
            else
                rounded = -(((-wide) + half) >>> shift);
            if (rounded > 32767)
                requantize = {1'b1, 16'h7fff};
            else if (rounded < -32768)
                requantize = {1'b1, 16'h8000};
            else
                requantize = {1'b0, rounded[15:0]};
        end
    endfunction

    function [10:0] probability_address;
        input signed [15:0] first, second;
        input [4:0] fraction;
        reg signed [31:0] difference, offset, half;
        begin
            difference = $signed({second[15], second}) - $signed({first[15], first});
            if (fraction > 6) begin
                half = 32'sd1 << (fraction-7);
                if (difference >= 0)
                    offset = (difference + half) >>> (fraction-6);
                else
                    offset = -(((-difference) + half) >>> (fraction-6));
            end else begin
                offset = difference <<< (6-fraction);
            end
            offset = offset + 512;
            if (offset < 0) probability_address = 0;
            else if (offset > 1024) probability_address = 1024;
            else probability_address = offset[10:0];
        end
    endfunction

    always @* begin
        read_address = group_index*960 + position;
        input_inside = 1;
        input_lane = term[2:0];
        neighbor_row = row;
        neighbor_column = column;
        if (kernel) begin
            neighbor_row = row + (term / 3) - 1;
            neighbor_column = column + (term % 3) - 1;
            input_inside = neighbor_row >= 0 && neighbor_row < 40 && neighbor_column >= 0 && neighbor_column < 24;
            if (input_inside)
                read_address = (depthwise ? group_index*960 : 0) + neighbor_row*24 + neighbor_column;
            else
                read_address = 0;
            input_lane = 0;
        end else if (state != POOL_READ && state != POOL_ADD && state != POOL_INIT) begin
            read_address = (term / 8)*960 + position;
        end
    end

    always @* begin
        case (output_index)
            0: m_axis_tdata = 32'h314e4e43;
            1: m_axis_tdata = 32'h00010008;
            2: m_axis_tdata = frame_id;
            3: m_axis_tdata = {{16{score0[15]}}, score0};
            4: m_axis_tdata = {{16{score1[15]}}, score1};
            5: m_axis_tdata = {flags, probability};
            6: m_axis_tdata = cycles;
            7: m_axis_tdata = configuration[8];
            default: m_axis_tdata = 0;
        endcase
    end

    genvar lane;
    generate for (lane = 0; lane < 8; lane = lane + 1) begin: lanes
        localparam [7:0] NUMBER = 48 + lane;
        (* ram_style = "block" *) reg signed [15:0] bank_a [0:4095];
        (* ram_style = "block" *) reg signed [15:0] bank_b [0:4095];
        (* rom_style = "distributed" *) reg signed [7:0] weights [0:255];
        (* rom_style = "distributed" *) reg signed [47:0] biases [0:15];
        reg signed [15:0] read_a, read_b, quantized;
        reg signed [47:0] accumulator;
        reg signed [31:0] pool_sum;
        wire signed [15:0] selected = depthwise ? $signed(read_value[lane]) : $signed(read_value[sampled_lane]);
        wire signed [15:0] operand = layer == 5 ? pooled[term] : sampled_inside ? selected : 16'sd0;
        wire signed [23:0] product = operand * weights[weight_address];
        wire [16:0] scaled = requantize(accumulator, configuration[layer+1][5:0], layer != 5);
        wire signed [31:0] pool_total = pool_sum + $signed(read_value[lane]);
        assign read_value[lane] = source_bank ? read_b : read_a;
        assign stored_value[lane] = quantized;
        assign quant_clips[lane] = scaled[16];
        assign pool_value[lane] = pool_total;

        initial begin
            $readmemh({MODEL_DIR, "/weights", NUMBER, ".mem"}, weights);
            $readmemh({MODEL_DIR, "/bias", NUMBER, ".mem"}, biases);
        end

        // Channel modulo eight selects the bank; each RAM has one read and one write port.
        always @(posedge clk) begin
            if (resetn && load_enable && lane == 0)
                bank_a[input_index] <= s_axis_tdata[15:0];
            else if (resetn && state == STORE && layer != 5 && source_bank)
                bank_a[write_address] <= quantized;
            if (resetn && state == STORE && layer != 5 && !source_bank)
                bank_b[write_address] <= quantized;
            if (resetn && read_enable) begin
                read_a <= bank_a[read_address];
                read_b <= bank_b[read_address];
            end
        end

        always @(posedge clk or negedge resetn) begin
            if (!resetn) begin
                accumulator <= 0;
                quantized <= 0;
                pool_sum <= 0;
            end else begin
                if (state == SETUP)
                    accumulator <= biases[bias_base+group_index];
                if (state == MAC)
                    accumulator <= accumulator + {{24{product[23]}}, product};
                if (state == QUANT)
                    quantized <= scaled[15:0];
                if (state == POOL_INIT)
                    pool_sum <= 0;
                if (state == POOL_ADD)
                    pool_sum <= pool_total;
            end
        end
    end endgenerate

    integer index;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state <= LOAD;
            layer <= 0;
            source_bank <= 0;
            group_index <= 0;
            position <= 0;
            row <= 0;
            column <= 0;
            term <= 0;
            input_index <= 0;
            input_bad <= 0;
            sampled_inside <= 0;
            sampled_lane <= 0;
            output_index <= 0;
            next_id <= 0;
            frame_id <= 0;
            cycles <= 0;
            flags <= 0;
            score0 <= 0;
            score1 <= 0;
            probability <= 0;
        end else begin
            if (state != LOAD && state != DRAIN && state != OUTPUT)
                cycles <= cycles + 1'b1;
            case (state)
                LOAD: if (s_axis_tvalid) begin
                    if (input_index == 0) begin
                        frame_id <= next_id;
                        next_id <= next_id + 1'b1;
                        flags <= 0;
                        cycles <= 0;
                    end
                    input_bad <= input_bad || bad_word;
                    if (s_axis_tlast && (input_index != 959 || input_bad || bad_word)) begin
                        score0 <= 0; score1 <= 0; probability <= 0;
                        flags <= 16'h0002; output_index <= 0; state <= OUTPUT;
                    end else if (input_index == 959) begin
                        if (!s_axis_tlast) begin
                            input_bad <= 1;
                            state <= DRAIN;
                        end else begin
                            layer <= 0; source_bank <= 0; group_index <= 0;
                            position <= 0; row <= 0; column <= 0;
                            state <= SETUP;
                        end
                    end else begin
                        input_index <= input_index + 1'b1;
                    end
                end
                DRAIN: if (s_axis_tvalid && s_axis_tlast) begin
                    score0 <= 0; score1 <= 0; probability <= 0;
                    flags <= 16'h0002; output_index <= 0; state <= OUTPUT;
                end
                SETUP: begin term <= 0; state <= READ_MAC; end
                READ_MAC: begin
                    sampled_inside <= input_inside;
                    sampled_lane <= input_lane;
                    state <= MAC;
                end
                MAC: if (term == terms-1) begin state <= QUANT; end
                     else begin term <= term + 1'b1; state <= READ_MAC; end
                QUANT: begin
                    if (|(quant_clips & active_mask)) flags[0] <= 1;
                    state <= STORE;
                end
                STORE: begin
                    if (layer == 5) begin
                        score0 <= stored_value[0]; score1 <= stored_value[1];
                        state <= PROB;
                    end else if (position != 959) begin
                        position <= position + 1'b1;
                        if (column == 23) begin column <= 0; row <= row + 1'b1; end
                        else column <= column + 1'b1;
                        state <= SETUP;
                    end else if (group_index != final_group) begin
                        group_index <= group_index + 1'b1;
                        position <= 0; row <= 0; column <= 0;
                        state <= SETUP;
                    end else begin
                        position <= 0; row <= 0; column <= 0; group_index <= 0;
                        source_bank <= !source_bank;
                        if (layer == 4) state <= POOL_INIT;
                        else begin layer <= layer + 1'b1; state <= SETUP; end
                    end
                end
                POOL_INIT: state <= POOL_READ;
                POOL_READ: state <= POOL_ADD;
                POOL_ADD: begin
                    if (position == 959) begin
                        for (index = 0; index < 8; index = index + 1)
                            pooled[group_index*8+index] <= (pool_value[index] + 480) / 960;
                        position <= 0;
                        if (group_index == 3) begin
                            group_index <= 0; layer <= 5; state <= SETUP;
                        end else begin
                            group_index <= group_index + 1'b1; state <= POOL_INIT;
                        end
                    end else begin position <= position + 1'b1; state <= POOL_READ; end
                end
                PROB: begin
                    probability <= sigmoid[probability_address(score0, score1, configuration[7][4:0])];
                    output_index <= 0; state <= OUTPUT;
                end
                OUTPUT: if (m_axis_tready) begin
                    if (output_index == 7) begin
                        input_index <= 0; input_bad <= 0; state <= LOAD;
                    end else output_index <= output_index + 1'b1;
                end
                default: state <= LOAD;
            endcase
        end
    end
endmodule
