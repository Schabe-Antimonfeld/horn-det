`timescale 1ns / 1ps

module pdm_capture #(
    parameter PAIRS = 7,
    parameter HOP = 512,
    parameter WARMUP_BITS = 153600
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK",
       X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF M_AXIS, ASSOCIATED_RESET resetn, FREQ_HZ 12288000" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 control_clk CLK",
       X_INTERFACE_PARAMETER = "FREQ_HZ 100000000" *)
    input wire control_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire resetn,
    input wire enable,
    input wire [PAIRS-1:0] pdm_data_i,
    output wire pdm_clk_o,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast,
    output wire [31:0] status
);
    wire [2*PAIRS-1:0] pdm_samples;
    wire pdm_valid;
    wire [32*PAIRS-1:0] pcm_samples;
    wire pcm_valid, clipped;
    wire [31:0] dropped;
    reg [31:0] warmup;
    (* ASYNC_REG = "TRUE" *) reg enable_meta, enable_sync;
    reg clip_seen;
    (* ASYNC_REG = "TRUE" *) reg [31:0] status_meta, status_sync;
    wire ready = warmup == WARMUP_BITS;

    assign status = status_sync;

    // Counters here are diagnostic snapshots; packet headers are authoritative.
    always @(posedge control_clk or negedge resetn) begin
        if (!resetn) begin
            status_meta <= 0;
            status_sync <= 0;
        end else begin
            status_meta <= {dropped[23:0], 5'd0, clip_seen, ready, enable_sync};
            status_sync <= status_meta;
        end
    end

    pdm_rx #(.PAIRS(PAIRS)) receiver (
        .clk(clk), .resetn(resetn), .pdm_data(pdm_data_i),
        .pdm_clock(pdm_clk_o), .samples(pdm_samples), .valid(pdm_valid)
    );
    pdm_decimator #(.CHANNELS(2*PAIRS)) decimator (
        .clk(clk), .resetn(resetn), .pdm_samples(pdm_samples), .pdm_valid(pdm_valid),
        .pcm_samples(pcm_samples), .pcm_valid(pcm_valid), .clipped(clipped)
    );
    axis_pcm #(.CHANNELS(2*PAIRS), .HOP(HOP)) packets (
        .clk(clk), .resetn(resetn), .enable(enable_sync && ready),
        .samples(pcm_samples), .valid(pcm_valid && ready), .clipped(clipped),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast), .dropped(dropped)
    );

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            warmup <= 0;
            enable_meta <= 0;
            enable_sync <= 0;
            clip_seen <= 0;
        end else begin
            enable_meta <= enable;
            enable_sync <= enable_meta;
            if (pdm_valid && !ready)
                warmup <= warmup + 1'b1;
            if (clipped)
                clip_seen <= 1;
        end
    end
endmodule
