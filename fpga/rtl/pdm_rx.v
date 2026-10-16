`timescale 1ns / 1ps

module pdm_rx #(
    parameter PAIRS = 7
) (
    input wire clk,
    input wire resetn,
    input wire [PAIRS-1:0] pdm_data,
    output wire pdm_clock,
    output reg [2*PAIRS-1:0] samples,
    output reg valid
);
    reg [1:0] phase;
    reg primed;
    wire [PAIRS-1:0] rising_data, falling_data;
    integer channel;

`ifdef SYNTHESIS
    BUFG pdm_buffer (.I(phase[1]), .O(pdm_clock));
    genvar pair_index;
    generate for (pair_index = 0; pair_index < PAIRS; pair_index = pair_index + 1) begin: inputs
        IDDR #(.DDR_CLK_EDGE("OPPOSITE_EDGE"), .SRTYPE("ASYNC")) input_ddr (
            .C(pdm_clock), .CE(1'b1), .D(pdm_data[pair_index]),
            .R(!resetn), .S(1'b0),
            .Q1(rising_data[pair_index]), .Q2(falling_data[pair_index])
        );
    end endgenerate
`else
    // Functional primitive substitutes; vendor timing simulation is still required.
    reg [PAIRS-1:0] rising_model, falling_model;
    assign pdm_clock = phase[1];
    assign rising_data = rising_model;
    assign falling_data = falling_model;
    always @(posedge pdm_clock or negedge resetn)
        if (!resetn) rising_model <= 0;
        else rising_model <= pdm_data;
    always @(negedge pdm_clock or negedge resetn)
        if (!resetn) falling_model <= 0;
        else falling_model <= pdm_data;
`endif

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            phase <= 0;
            primed <= 0;
            samples <= 0;
            valid <= 0;
        end else begin
            phase <= phase + 1'b1;
            valid <= 0;
            if (phase == 3)
                primed <= 1;
            if (phase == 0 && primed) begin
                for (channel = 0; channel < PAIRS; channel = channel + 1) begin
                    samples[2*channel] <= rising_data[channel];
                    samples[2*channel+1] <= falling_data[channel];
                end
                valid <= 1;
            end
        end
    end
endmodule
