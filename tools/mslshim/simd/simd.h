// Stand-in for Apple's <simd/simd.h> when type-checking .metal files with tools/check_msl.py.
#pragma once
#include <metal_stdlib>
typedef metal::float2 vector_float2;
typedef metal::float3 vector_float3;
typedef metal::float4 vector_float4;
typedef metal::int2 vector_int2;
typedef metal::int3 vector_int3;
typedef metal::int4 vector_int4;
typedef metal::uint2 vector_uint2;
typedef metal::uint3 vector_uint3;
typedef metal::uint4 vector_uint4;
typedef metal::float2x2 matrix_float2x2;
typedef metal::float3x3 matrix_float3x3;
typedef metal::float4x4 matrix_float4x4;
