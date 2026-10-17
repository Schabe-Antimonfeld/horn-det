`timescale 1ns/1ps

// A DMA packet contains a direction count followed by PAIRS triples per direction.
module srp_batch #(parameter PAIRS = 91, parameter MAX_DIRECTIONS = 128) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET resetn" *)
    input wire clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *) input wire resetn,
    input wire [31:0] s_axis_tdata, input wire [3:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready, input wire s_axis_tlast,
    output wire [31:0] m_axis_tdata, output wire [3:0] m_axis_tkeep,
    output wire m_axis_tvalid, input wire m_axis_tready, output wire m_axis_tlast
);
    localparam HEADER = 0, INPUTS = 1, OUTPUTS = 2, INVALID = 3;
    reg [1:0] state;
    integer remaining, words;
    wire core_ready, core_valid, core_last;
    wire ending = words == PAIRS*3-1;
    wire expected_last = ending && remaining == 1;
    wire bad_last = s_axis_tlast != expected_last;
    wire malformed = s_axis_tkeep != 4'hf || s_axis_tlast ||
                     s_axis_tdata == 0 || s_axis_tdata > MAX_DIRECTIONS;
    assign s_axis_tready = state == HEADER || (state == INPUTS && core_ready);
    assign m_axis_tvalid = state == OUTPUTS && core_valid;
    assign m_axis_tlast = m_axis_tvalid && core_last && remaining == 1;
    srp_accumulator #(.PAIRS(PAIRS)) kernel (
        .clk(clk), .resetn(resetn),
        .s_axis_tdata(state == INVALID ? 32'b0 : s_axis_tdata),
        .s_axis_tkeep(state == INVALID || bad_last ? 4'b0 : s_axis_tkeep),
        .s_axis_tvalid(state == INVALID || (state == INPUTS && s_axis_tvalid)),
        .s_axis_tready(core_ready),
        .s_axis_tlast(state == INVALID || ending || s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(core_valid), .m_axis_tready(state == OUTPUTS && m_axis_tready),
        .m_axis_tlast(core_last)
    );
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin state <= HEADER; remaining <= 0; words <= 0; end
        else begin
            if (state == HEADER && s_axis_tvalid) begin
                remaining <= malformed ? 1 : s_axis_tdata;
                words <= 0;
                state <= malformed ? INVALID : INPUTS;
            end
            if (state == INVALID && core_ready) state <= OUTPUTS;
            if (state == INPUTS && s_axis_tvalid && core_ready) begin
                words <= words + 1;
                if (ending || s_axis_tlast) begin
                    if (s_axis_tlast && !expected_last) remaining <= 1;
                    state <= OUTPUTS;
                end
            end
            if (state == OUTPUTS && core_valid && m_axis_tready && core_last) begin
                words <= 0;
                remaining <= remaining - 1;
                state <= remaining == 1 ? HEADER : INPUTS;
            end
        end
    end
endmodule
