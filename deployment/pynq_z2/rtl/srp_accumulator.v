`timescale 1ns/1ps

// Optional scoring kernel: each pair supplies lower, upper and Q0.16 fraction.
module srp_accumulator #(parameter PAIRS = 91) (
    input wire clk, input wire resetn,
    input wire [31:0] s_axis_tdata, input wire [3:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready, input wire s_axis_tlast,
    output reg [31:0] m_axis_tdata, output wire [3:0] m_axis_tkeep,
    output wire m_axis_tvalid, input wire m_axis_tready, output wire m_axis_tlast
);
    reg [1:0] phase, output_index;
    reg signed [23:0] lower, upper;
    reg signed [47:0] total, result;
    reg pending, failed, output_error;
    integer count;
    wire signed [17:0] first_weight = 18'sd65536 - $signed({2'b00, s_axis_tdata[15:0]});
    wire signed [17:0] second_weight = $signed({2'b00, s_axis_tdata[15:0]});
    wire signed [41:0] first_term = lower * first_weight;
    wire signed [41:0] second_term = upper * second_weight;
    wire signed [47:0] interpolated = {{6{first_term[41]}}, first_term} + {{6{second_term[41]}}, second_term};
    wire invalid = s_axis_tkeep != 4'hf ||
        (phase != 2 && s_axis_tdata[31:24] != {8{s_axis_tdata[23]}}) ||
        (phase == 2 && s_axis_tdata[31:16] != 0);
    assign s_axis_tready = !pending;
    assign m_axis_tvalid = pending;
    assign m_axis_tkeep = 4'hf;
    assign m_axis_tlast = pending && output_index == 3;
    always @* begin
        case (output_index)
            0: m_axis_tdata = 32'h31505253;
            1: m_axis_tdata = result[31:0];
            2: m_axis_tdata = {{16{result[47]}}, result[47:32]};
            default: m_axis_tdata = {31'b0, output_error};
        endcase
    end
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            phase <= 0; output_index <= 0; lower <= 0; upper <= 0;
            total <= 0; result <= 0; pending <= 0; failed <= 0; output_error <= 0; count <= 0;
        end else begin
            if (pending && m_axis_tready) begin
                if (output_index == 3) begin pending <= 0; output_index <= 0; end
                else output_index <= output_index + 1'b1;
            end
            if (s_axis_tvalid && s_axis_tready) begin
                if (invalid) failed <= 1;
                if (phase == 0) lower <= s_axis_tdata[23:0];
                if (phase == 1) upper <= s_axis_tdata[23:0];
                if (phase == 2) begin
                    total <= total + interpolated;
                    if (count < PAIRS) count <= count + 1;
                    else failed <= 1;
                end
                if (s_axis_tlast) begin
                    output_error <= failed || invalid || phase != 2 || count != PAIRS-1;
                    result <= (failed || invalid || phase != 2 || count != PAIRS-1) ? 0 : total + interpolated;
                    pending <= 1; output_index <= 0;
                    phase <= 0; count <= 0; total <= 0; failed <= 0;
                end else phase <= phase == 2 ? 0 : phase + 1'b1;
            end
        end
    end
endmodule
