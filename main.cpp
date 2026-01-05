#include <bits/stdc++.h>
#include <x86intrin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
using namespace std;

static double dequantise_naive(uint8_t val, int bias = -9) {
    // Bitmasks for projecting out mantissa, exponent, sign
    static constexpr uint8_t mntmask = 15;   // 0b00001111
    static constexpr uint8_t expmask = 112;  // 0b01110000
    static constexpr uint8_t sgnmask = 128;  // 0b10000000
    // Apply masks and shift to decode components
    uint8_t mnt = val & mntmask;
    uint8_t exp = (val & expmask) >> 4;
    const double sgn = (val & sgnmask) ? -1.0 : +1.0;
    // Leading digit in mantissa is implicit, unless zero exponent
    mnt = exp ? (16 | mnt) : mnt;
    exp = exp ? exp : 1;
    // Calculate and return value
    return sgn * ((double)mnt) * pow(2, (int)exp + bias);
}

void fill_argmax_naive(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
        int64_t max_idx = 0;
        double max_val = std::numeric_limits<double>::min();

        for (int64_t j = 0; j < 1024; j++) {
            uint8_t qres = _buffer[1024 * i + j];
            double res = dequantise_naive(qres);
            if (res > max_val) {
                max_idx = j;
                max_val = res;
            }
        }

        bufs[i] = max_idx;
    }
}

static float dequantise(uint8_t val) {
    // Bitmasks for projecting out mantissa, exponent, sign
    static constexpr uint8_t mntmask = 15;   // 0b00001111
    static constexpr uint8_t expmask = 112;  // 0b01110000
    static constexpr uint8_t sgnmask = 128;  // 0b10000000
    // Apply masks and shift to decode components
    uint8_t mnt = val & mntmask;
    uint8_t exp = (val & expmask) >> 4;
    const float sgn = (val & sgnmask) ? -1.0 : +1.0;
    // Leading digit in mantissa is implicit, unless zero exponent
    mnt = exp ? (16 | mnt) : mnt;
    exp = exp ? exp : 1;
    // Calculate and return value
    return sgn * (float)(mnt << (int)exp);
}

void fill_argmax(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
        int64_t max_idx = 0;
        float max_val = std::numeric_limits<float>::min();

        for (int64_t j = 0; j < 1024; j++) {
            uint8_t qres = _buffer[1024 * i + j];
            float res = dequantise(qres);
            if (res > max_val) {
                max_idx = j;
                max_val = res;
            }
        }

        bufs[i] = max_idx;
    }
}

static bool fp8_lt(int8_t a, int8_t b) {
    // Bitmasks for projecting out mantissa, exponent, sign
    static constexpr uint8_t mntmask = 15;   // 0b00001111
    static constexpr uint8_t expmask = 112;  // 0b01110000
    static constexpr uint8_t sgnmask = 128;  // 0b10000000

    uint8_t a_mnt = a & mntmask;
    uint8_t a_exp = a & expmask;
    uint8_t a_sgn = a & sgnmask;
    a_mnt = a_exp ? (16 | a_mnt) : a_mnt;
    a_exp = a_exp ? a_exp : 1;

    uint8_t b_mnt = b & mntmask;
    uint8_t b_exp = b & expmask;
    uint8_t b_sgn = b & sgnmask;
    b_mnt = b_exp ? (16 | b_mnt) : b_mnt;
    b_exp = b_exp ? b_exp : 1;

    if (a_sgn != b_sgn) {
        return a_sgn > 0;
    }
    if (a_exp != b_exp) {
        return (a_sgn > 0) ^ (a_exp < b_exp);
    }
    if (a_mnt != b_mnt) {
        return (a_sgn > 0) ^ (a_mnt < b_mnt);
    }
    return false;
}

void fill_argmax_fp8_lt(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
        int64_t max_idx = 0;
        int8_t max_val = -127;

        for (int64_t j = 0; j < 1024; j++) {
            int8_t qres = _buffer[1024 * i + j];
            if (fp8_lt(max_val, qres)) {
                max_idx = j;
                max_val = qres;
            }
        }

        bufs[i] = max_idx;
    }
}

void fast_fp8_fused_dequantise_argmax_uint(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
        int64_t max_idx = 0;
        uint8_t min_val = 255;

        for (int64_t j = 0; j < 1024; j++) {
            uint8_t qres = _buffer[1024 * i + j];

            if (qres < 128) {
                qres = qres ^ 0b01111111;
            }

            if (qres < min_val) {
                max_idx = j;
                min_val = qres;
            }
        }

        bufs[i] = max_idx;
    }
}

void fast_fp8_fused_dequantise_argmax_int(
    const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
        int64_t max_idx = 0;
        int8_t max_val = -128;

        for (int64_t j = 0; j < 1024; j++) {
            int8_t qres = (int8_t)_buffer[1024 * i + j];

            if (qres < 0) {
                qres = qres ^ 0b01111111;
            }

            if (qres > max_val) {
                max_idx = j;
                max_val = qres;
            }
        }
        bufs[i] = max_idx;
    }
}

int8_t int_deq(const int8_t qres) {
    if (qres < 0) {
        return qres ^ 0b01111111;
    }
    return qres;
}

void simd_fast_fp8_fused_dequantise_argmax(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    // Constants used to "dequantise" the quantized values
    const __m256i boundary = _mm256_setzero_si256();
    const __m256i xor_mask = _mm256_set1_epi8(0b01111111);
    const __m256i plus32 = _mm256_set1_epi8(1);  // Remember to convert this back to 32

    for (size_t i = 0; i < bufs.size(); i++) {
        // Track the current values' indices
        // But an 8 bit integer can only represent indices upto 127, whilst our indices can reach 1023
        // To fix this, we'll use 2 indices, the 32 byte index, and an offset into the 32 byte vector
        // idx stores the 32 byte index, and the position of an index within idx, represents the offset
        // So to get the actual index, we need idx[k] * 32 + k
        __m256i idx = _mm256_setzero_si256();

        // Track the maximum values and their indices - indices are as above
        __m256i max_val_v = _mm256_set1_epi8(-128);
        __m256i max_idx_v = _mm256_setzero_si256();

        for (size_t j = 0; j < 1024; j += 32) {
            // Load 32 values from the input buffer into a 256-bit vector.
            const __m256i* src_ptr = reinterpret_cast<const __m256i*>(_buffer + 1024 * i + j);
            __m256i qres_v = _mm256_stream_load_si256(src_ptr);

            // XOR all values that are less than 0 with 0b0111_1111.
            // Now if v1 < v2 then deQuant(v1) < deQuant(v2).
            __m256i mask = _mm256_cmpgt_epi8(boundary, qres_v);
            mask = _mm256_and_si256(mask, xor_mask);
            qres_v = _mm256_xor_si256(qres_v, mask);

            // Update the maximum values and their indices.
            const __m256i gt = _mm256_cmpgt_epi8(qres_v, max_val_v);
            max_idx_v = _mm256_blendv_epi8(max_idx_v, idx, gt);
            max_val_v = _mm256_max_epi8(max_val_v, qres_v);

            // Update all current index
            idx = _mm256_add_epi8(idx, plus32);
        }

        // Get the vector values out
        int8_t max_arr[32], idx_arr[32];
        _mm256_storeu_si256((__m256i*)max_arr, max_val_v);
        _mm256_storeu_si256((__m256i*)idx_arr, max_idx_v);

        // Argmax over the vector
        int8_t max_val = -128;
        size_t max_idx = 0;

        for (int64_t k = 0; k < 32; k++) {
            const size_t idx = idx_arr[k] * 32 + k;
            const int8_t val = max_arr[k];

            if (val > max_val) {
                max_idx = idx;
                max_val = val;
            }
        }
        bufs[i] = max_idx;
    }
}

#ifdef __AVX512F__
void simd_fast_fp8_fused_dequantise_argmax512(const uint8_t* _buffer, std::vector<int64_t>& bufs) {  //}, int64_t timings[3]) {
    // Constants used to "dequantise" the quantized values
    const __m512i boundary = _mm512_setzero_si512();
    const __m512i xor_mask = _mm512_set1_epi8(0b01111111);
    const __m512i plus64 = _mm512_set1_epi8(1);  // Remember to convert this back to 64

    // int64_t load_time = 0;
    // int64_t process_time = 0;
    // int64_t store_time = 0;

    for (size_t i = 0; i < bufs.size(); i++) {
        // Track the current values' indices
        // But an 8 bit integer can only represent indices upto 127, whilst our indices can reach 1023
        // To fix this, we'll use 2 indices, the 64 byte index, and an offset into the 64 byte vector
        // idx stores the 64 byte index, and the position of an index within idx, represents the offset
        // So to get the actual index, we need idx[k] * 64 + k
        __m512i idx = _mm512_setzero_si512();

        // Track the maximum values and their indices - indices are as above
        __m512i max_val_v = _mm512_set1_epi8(-128);
        __m512i max_idx_v = _mm512_setzero_si512();

        for (size_t j = 0; j < 1024; j += 64) {
            // Load 32 values from the input buffer into a 512-bit vector.
            const void* src_ptr = (_buffer + 1024 * i + j);
            // auto s1 = std::chrono::high_resolution_clock::now();
            __m512i qres_v = _mm512_stream_load_si512((void*)src_ptr);
            // auto s2 = std::chrono::high_resolution_clock::now();
            // load_time += (s2 - s1).count();

            // XOR all values that are less than 0 with 0b0111_1111.
            // Now if v1 < v2 then deQuant(v1) < deQuant(v2).
            __mmask64 k = _mm512_cmpgt_epi8_mask(boundary, qres_v);
            __m512i mask = _mm512_movm_epi8(k);
            mask = _mm512_and_si512(mask, xor_mask);
            qres_v = _mm512_xor_si512(qres_v, mask);

            // Update the maximum values and their indices.
            const __mmask64 gt = _mm512_cmpgt_epi8_mask(qres_v, max_val_v);
            max_idx_v = _mm512_mask_blend_epi8(gt, max_idx_v, idx);
            max_val_v = _mm512_max_epi8(max_val_v, qres_v);

            // Update all current index
            idx = _mm512_add_epi8(idx, plus64);
            // auto s3 = std::chrono::high_resolution_clock::now();
            // process_time += (s3 - s2).count();
        }

        // Get the vector values out
        int8_t max_arr[64], idx_arr[64];
        // auto s1 = std::chrono::high_resolution_clock::now();
        _mm512_storeu_si512((__m512i*)max_arr, max_val_v);
        _mm512_storeu_si512((__m512i*)idx_arr, max_idx_v);
        // auto s2 = std::chrono::high_resolution_clock::now();

        // store_time += (s2 - s1).count();

        // Argmax over the vector
        int8_t max_val = -128;
        size_t max_idx = 0;

        for (int64_t k = 0; k < 64; k++) {
            const size_t idx = idx_arr[k] * 64 + k;
            const int8_t val = max_arr[k];

            if (val > max_val) {
                max_idx = idx;
                max_val = val;
            }
        }
        bufs[i] = max_idx;
    }
    // timings[0] = load_time / bufs.size();
    // timings[1] = process_time / bufs.size();
    // timings[2] = store_time / bufs.size();
}
#endif

int main() {
    const size_t batch_size = 8;
    const size_t loops = 1000000;

    uint8_t* _buffer = reinterpret_cast<uint8_t*>(aligned_alloc(512, 1024 * batch_size));

    for (size_t i = 0; i < 1024 * batch_size; i++) {
        _buffer[i] = rand() % 50;
    }

    std::vector<int64_t> gts(batch_size);
    for (auto i = 0; i < batch_size; i++) {
        gts[i] = rand() % 1024;
        _buffer[i * 1024 + gts[i]] = 127;
    }

    std::vector<int64_t> bufs(batch_size);

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            fill_argmax_naive(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Naive " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            fill_argmax(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Dequant " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            fill_argmax_fp8_lt(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "fp8cmp " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            fast_fp8_fused_dequantise_argmax_uint(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "fast_fp8_fused_uint " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            fast_fp8_fused_dequantise_argmax_int(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "fast_fp8_fused_int " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < loops; i++) {
            simd_fast_fp8_fused_dequantise_argmax(_buffer, bufs);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "simd256 " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
    }
#ifdef __AVX512F__
    {
        int64_t timings[3] = {0, 0, 0};
        auto start = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < loops; i++) {
            // int64_t timings_temp[3] = {0, 0, 0};
            simd_fast_fp8_fused_dequantise_argmax512(_buffer, bufs);  //, timings);
            // timings[0] += timings_temp[0] / loops;
            // timings[2] += timings_temp[1] / loops;
            // timings[1] += timings_temp[2] / loops;
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "simd512 " << (end - start).count() / loops << "ns" << std::endl;
        for (auto i = 0; i < batch_size; i++) {
            std::cout << bufs[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "\t";
        for (auto i = 0; i < 3; i++) {
            std::cout << timings[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif
    free(_buffer);
}
