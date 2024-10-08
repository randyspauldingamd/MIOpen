#pragma once

struct poolingNdNhwcArgs
{
    uint32_t filter_d;
    uint32_t filter_h;
    uint32_t filter_w;

    uint32_t filter_d_stride;
    uint32_t filter_h_stride;
    uint32_t filter_w_stride;

    uint32_t filter_d_pad;
    uint32_t filter_h_pad;
    uint32_t filter_w_pad;

    uint32_t all_n;
    uint32_t all_c;

    uint32_t bot_d;
    uint32_t bot_h;
    uint32_t bot_w;

    uint32_t bot_n_stride;
    uint32_t bot_c_stride;
    uint32_t bot_d_stride;
    uint32_t bot_h_stride;
    uint32_t bot_w_stride;

    uint32_t top_d;
    uint32_t top_h;
    uint32_t top_w;

    uint32_t top_n_stride;
    uint32_t top_c_stride;
    uint32_t top_d_stride;
    uint32_t top_h_stride;
    uint32_t top_w_stride;

    uint32_t mask_n_stride;
    uint32_t mask_c_stride;
    uint32_t mask_d_stride;
    uint32_t mask_h_stride;
    uint32_t mask_w_stride;
};
