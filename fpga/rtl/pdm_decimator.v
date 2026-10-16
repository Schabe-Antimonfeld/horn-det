`timescale 1ns / 1ps

module pdm_decimator #(
    parameter CHANNELS = 14
) (
    input wire clk,
    input wire resetn,
    input wire [CHANNELS-1:0] pdm_samples,
    input wire pdm_valid,
    output wire [16*CHANNELS-1:0] pcm_samples,
    output reg pcm_valid,
    output wire clipped
);
    reg [4:0] divider;
    reg cic_valid;
    reg fir_phase;
    reg [5:0] write_index;
    reg [6:0] filled;
    reg [6:0] used;
    reg [5:0] read_index;
    reg [5:0] tap;
    reg busy;
    wire start_fir = cic_valid && fir_phase;
    wire [CHANNELS-1:0] clips;

    function signed [17:0] coefficient;
        input [5:0] index;
        begin
            case (index)
                0,62: coefficient = -18'sd1;
                1,61: coefficient = -18'sd9;
                2,60: coefficient = -18'sd21;
                3,59: coefficient = -18'sd12;
                4,58: coefficient = 18'sd37;
                5,57: coefficient = 18'sd73;
                6,56: coefficient = 18'sd1;
                7,55: coefficient = -18'sd146;
                8,54: coefficient = -18'sd142;
                9,53: coefficient = 18'sd139;
                10,52: coefficient = 18'sd372;
                11,51: coefficient = 18'sd79;
                12,50: coefficient = -18'sd552;
                13,49: coefficient = -18'sd573;
                14,48: coefficient = 18'sd420;
                15,47: coefficient = 18'sd1213;
                16,46: coefficient = 18'sd288;
                17,45: coefficient = -18'sd1599;
                18,44: coefficient = -18'sd1618;
                19,43: coefficient = 18'sd1141;
                20,42: coefficient = 18'sd3188;
                21,41: coefficient = 18'sd688;
                22,40: coefficient = -18'sd4094;
                23,39: coefficient = -18'sd4008;
                24,38: coefficient = 18'sd3003;
                25,37: coefficient = 18'sd8212;
                26,36: coefficient = 18'sd1774;
                27,35: coefficient = -18'sd11656;
                28,34: coefficient = -18'sd13156;
                29,33: coefficient = 18'sd9451;
                30,32: coefficient = 18'sd43270;
                31: coefficient = 18'sd59548;
                default: coefficient = 0;
            endcase
        end
    endfunction

    function signed [23:0] cic_scale;
        input signed [31:0] value;
        reg signed [32:0] wide;
        reg signed [32:0] rounded;
        begin
            wide = value;
            if (wide >= 0)
                rounded = (wide + 33'sd2) >>> 2;
            else
                rounded = -(((-wide) + 33'sd2) >>> 2);
            if (rounded > 33'sd8388607)
                cic_scale = 24'sh7fffff;
            else if (rounded < -33'sd8388608)
                cic_scale = 24'sh800000;
            else
                cic_scale = rounded[23:0];
        end
    endfunction

    function [16:0] pcm_scale;
        input signed [47:0] value;
        reg signed [48:0] wide;
        reg signed [48:0] rounded;
        begin
            wide = value;
            if (wide >= 0)
                rounded = (wide + 49'sd16777216) >>> 25;
            else
                rounded = -(((-wide) + 49'sd16777216) >>> 25);
            if (rounded > 49'sd32767)
                pcm_scale = {1'b1, 16'h7fff};
            else if (rounded < -49'sd32768)
                pcm_scale = {1'b1, 16'h8000};
            else
                pcm_scale = {1'b0, rounded[15:0]};
        end
    endfunction

    assign clipped = |clips;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            divider <= 0;
            cic_valid <= 0;
            fir_phase <= 0;
            write_index <= 0;
            filled <= 0;
            used <= 0;
            read_index <= 0;
            tap <= 0;
            busy <= 0;
            pcm_valid <= 0;
        end else begin
            cic_valid <= pdm_valid && divider == 31;
            pcm_valid <= 0;
            if (pdm_valid)
                divider <= divider + 1'b1;
            if (cic_valid) begin
                write_index <= write_index + 1'b1;
                if (filled < 63)
                    filled <= filled + 1'b1;
                fir_phase <= !fir_phase;
                if (fir_phase) begin
                    busy <= 1;
                    tap <= 0;
                    read_index <= write_index;
                    used <= filled < 63 ? filled + 1'b1 : 7'd63;
                end
            end
            if (busy) begin
                if (tap == 62) begin
                    busy <= 0;
                    pcm_valid <= 1;
                end else begin
                    tap <= tap + 1'b1;
                end
            end
        end
    end

    genvar channel;
    generate for (channel = 0; channel < CHANNELS; channel = channel + 1) begin: lanes
        reg [31:0] integral0, integral1, integral2, integral3, integral4;
        reg [31:0] delay0, delay1, delay2, delay3, delay4;
        wire [31:0] next0 = integral0 + (pdm_samples[channel] ? 32'd1 : 32'hffffffff);
        wire [31:0] next1 = integral1 + next0;
        wire [31:0] next2 = integral2 + next1;
        wire [31:0] next3 = integral3 + next2;
        wire [31:0] next4 = integral4 + next3;
        wire [31:0] comb0 = next4 - delay0;
        wire [31:0] comb1 = comb0 - delay1;
        wire [31:0] comb2 = comb1 - delay2;
        wire [31:0] comb3 = comb2 - delay3;
        wire [31:0] comb4 = comb3 - delay4;
        reg signed [23:0] cic_sample;
        (* ram_style = "distributed" *) reg signed [23:0] history [0:63];
        wire [5:0] address = read_index - tap;
        wire signed [23:0] value = tap < used ? history[address] : 24'sd0;
        wire signed [41:0] product = value * coefficient(tap);
        reg signed [47:0] accumulator;
        wire signed [47:0] total = accumulator + {{6{product[41]}}, product};
        wire [16:0] result = pcm_scale(total);
        reg [15:0] pcm_sample;
        reg clip;

        assign pcm_samples[16*channel +: 16] = pcm_sample;
        assign clips[channel] = clip;

        // CIC integrators deliberately wrap; saturating them changes the filter.
        always @(posedge clk or negedge resetn) begin
            if (!resetn) begin
                integral0 <= 0; integral1 <= 0; integral2 <= 0;
                integral3 <= 0; integral4 <= 0;
                delay0 <= 0; delay1 <= 0; delay2 <= 0; delay3 <= 0; delay4 <= 0;
                cic_sample <= 0;
                accumulator <= 0;
                pcm_sample <= 0;
                clip <= 0;
            end else begin
                clip <= 0;
                if (pdm_valid) begin
                    integral0 <= next0; integral1 <= next1; integral2 <= next2;
                    integral3 <= next3; integral4 <= next4;
                    if (divider == 31) begin
                        delay0 <= next4; delay1 <= comb0; delay2 <= comb1;
                        delay3 <= comb2; delay4 <= comb3;
                        cic_sample <= cic_scale($signed(comb4));
                    end
                end
                if (start_fir)
                    accumulator <= 0;
                else if (busy) begin
                    accumulator <= total;
                    if (tap == 62) begin
                        pcm_sample <= result[15:0];
                        clip <= result[16];
                    end
                end
            end
        end

        // Filled-count masking makes old RAM contents irrelevant after reset.
        always @(posedge clk) begin
            if (resetn && cic_valid)
                history[write_index] <= cic_sample;
        end
    end endgenerate
endmodule
