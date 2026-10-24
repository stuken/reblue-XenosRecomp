#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

// PGR3 and PGR4 share Bizarre Creations' engine (gotham3, gotham4), and with it
// the vertex formats and fetch patterns GOTHAM_RECOMP handles. PGR3_RECOMP and
// PGR4_RECOMP keep the per-game tables (vertex input locations, interpolators).
#if (defined(PGR3_RECOMP) || defined(PGR4_RECOMP)) && !defined(GOTHAM_RECOMP)
#define GOTHAM_RECOMP
#endif

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#else
    // DEC3N/DEC4N tangent-basis inputs (k_2_10_10_10, signed normalised), as
    // opposed to the 11:11:10 packing SPEC_CONSTANT_R11G11B10_NORMAL decodes.
    // Only a GOTHAM_RECOMP recompiler sets it; PGR3 and PGR4 normals use it.
    #define SPEC_CONSTANT_DEC3N_NORMAL      (1 << 2)
    // The render target is Xenos' 7e3 EDRAM format, which cannot hold NaN,
    // infinity or values above 31.875: the pixel shader's colour is clamped
    // as the conversion to it does (PGR4_RECOMP pixel shaders).
    #define SPEC_CONSTANT_CLAMP_7E3         (1 << 3)
#endif

// SPIR-V vertex input locations, shared with reblue's host input-layout
// builder. A semantic missing here gets no location from the emitter.
#ifdef RR6_RECOMP
// Every semantic an RR6 vertex shader declares.
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,  0,  0) \
    X(Normal,    0,  1) \
    X(Tangent,   0,  2) \
    X(Tangent,   1,  3) \
    X(Binormal,  0,  4) \
    X(TexCoord,  0,  5) \
    X(TexCoord,  1,  6) \
    X(TexCoord,  2,  7) \
    X(Color,     0,  8)
#elif defined(PGR3_RECOMP)
// Every semantic a PGR3 vertex shader declares (21 across 105 shaders, none
// using more than 11). Locations run to 20, so Vulkan needs
// maxVertexInputAttributes >= 21; desktop drivers report 32 or more.
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,     0,  0) \
    X(Position,     1,  1) \
    X(Position,     2,  2) \
    X(Position,     3,  3) \
    X(BlendWeight,  0,  4) \
    X(BlendWeight,  1,  5) \
    X(BlendIndices, 0,  6) \
    X(Normal,       0,  7) \
    X(Normal,       1,  8) \
    X(Normal,       2,  9) \
    X(TexCoord,     0, 10) \
    X(TexCoord,     1, 11) \
    X(TexCoord,     2, 12) \
    X(TexCoord,     3, 13) \
    X(TexCoord,     4, 14) \
    X(TexCoord,     5, 15) \
    X(TexCoord,     6, 16) \
    X(Tangent,      0, 17) \
    X(Binormal,     0, 18) \
    X(Color,        0, 19) \
    X(Color,        1, 20)
#elif defined(PGR4_RECOMP)
// Every semantic a PGR4 vertex shader declares (23 across 364 shaders, none
// using more than 12). Locations run to 22, so Vulkan needs
// maxVertexInputAttributes >= 23; desktop drivers report 32 or more.
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,     0,  0) \
    X(Position,     1,  1) \
    X(Position,     2,  2) \
    X(Position,     3,  3) \
    X(Position,     4,  4) \
    X(Position,     5,  5) \
    X(Position,     6,  6) \
    X(BlendWeight,  0,  7) \
    X(BlendWeight,  1,  8) \
    X(BlendIndices, 0,  9) \
    X(Normal,       0, 10) \
    X(Normal,       1, 11) \
    X(Normal,       2, 12) \
    X(TexCoord,     0, 13) \
    X(TexCoord,     1, 14) \
    X(TexCoord,     2, 15) \
    X(TexCoord,     3, 16) \
    X(TexCoord,     4, 17) \
    X(TexCoord,     5, 18) \
    X(TexCoord,     6, 19) \
    X(Tangent,      0, 20) \
    X(Color,        0, 21) \
    X(Color,        1, 22)
#else
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,  0,  0) \
    X(Position,  1,  1) \
    X(Position,  2,  2) \
    X(Position,  3,  3) \
    X(Position,  4,  4) \
    X(Normal,    0,  5) \
    X(Tangent,   0,  6) \
    X(TexCoord,  0,  7) \
    X(TexCoord,  1,  8) \
    X(TexCoord,  2,  9) \
    X(Color,     0, 10)
#endif

#ifdef GOTHAM_RECOMP
// Vertex pulling. Ten PGR3 vertex shaders compute their own fetch index (per
// instance transforms at index / N, dependent fetches through a fetched
// index), which the input assembler cannot model, so the recompiler reads
// every input of such a shader from the raw stream buffers (pullVertex). The
// host describes each input location in SharedConstants as a uint2: x =
// stream | (PULL_FORMAT_* << 8), y = byte offset in the stream's element. The
// format codes name the host input-assembler format the declaration binds, so
// a pulled fetch decodes exactly as the input assembler would and the usual
// swapFloats/sintTexcoord/... wrappers apply unchanged. A stream at or above
// PULL_MAX_STREAMS reads as zero data.
#define PULL_MAX_STREAMS              4
#define PULL_MAX_LOCATIONS            22
#define PULL_FORMAT_NONE              0
#define PULL_FORMAT_R32_FLOAT         1
#define PULL_FORMAT_R32G32_FLOAT      2
#define PULL_FORMAT_R32G32B32_FLOAT   3
#define PULL_FORMAT_R32G32B32A32_FLOAT 4
#define PULL_FORMAT_R32_UINT          5
#define PULL_FORMAT_R32G32B32_UINT    6
#define PULL_FORMAT_B8G8R8A8_UNORM    7
#define PULL_FORMAT_R8G8B8A8_UNORM    8
#define PULL_FORMAT_R8G8B8A8_UINT     9
#define PULL_FORMAT_R16G16_SNORM      10
#define PULL_FORMAT_R16G16B16A16_SNORM 11
#define PULL_FORMAT_R16G16_UNORM      12
#define PULL_FORMAT_R16G16B16A16_UNORM 13
#define PULL_FORMAT_R16G16_SINT       14
#define PULL_FORMAT_R16G16B16A16_SINT 15
#define PULL_FORMAT_R16G16_UINT       16
#define PULL_FORMAT_R16G16B16A16_UINT 17
#define PULL_FORMAT_R16G16_FLOAT      18
#define PULL_FORMAT_R16G16B16A16_FLOAT 19
#endif

// SPEC_CONSTANTS_ONLY keeps the HLSL below out of host C++ TUs, which IntelliSense would otherwise parse.
#if (!defined(__cplusplus) || defined(__INTELLISENSE__)) && !defined(SHADER_COMMON_SPEC_CONSTANTS_ONLY)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

#ifdef REBLUE_RECOMP
// 256-bit boolean register file (BD bool addresses reach ~158), then per-usage 16-bit-pair swap masks.
#define g_Booleans(i)              vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256 + (i)*4)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 288)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 292)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 300)
#define g_SwappedNormals           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 304)
#define g_SwappedBinormals         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 308)
#define g_SwappedTangents          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 312)
#define g_SwappedBlendWeights      vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 316)
#define g_SwappedPositions         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 320)
#define g_SintTexcoords            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 324)
#define g_Uhend3nTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 348)
#define g_Hend3IntegerTexcoords    vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 472)
#define g_Hend3SignedTexcoords     vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 476)
#define g_ShadowPcfScale           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 328)
#define g_AlphaTestFunc            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 332)
#define g_ClipPlaneEnable          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 344)
#define g_ClipPlanes(i)            vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 352 + (i)*16)
#define g_NdcScale                 vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 448)
#define g_NdcOffset                vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 456)
#define g_EdgeSnap                 vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 464)
#ifdef GOTHAM_RECOMP
// VertexIndex already includes the draw's base vertex on Vulkan.
#define g_VertexIndexBase          0
#define g_UintTexcoords            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 484)
#define g_PulledStreamSizes        vk::RawBufferLoad<uint4>(g_PushConstants.SharedConstants + 496)
#define g_PulledStreamStrides      vk::RawBufferLoad<uint4>(g_PushConstants.SharedConstants + 512)
#define g_PulledStreamAddress(i)   vk::RawBufferLoad<uint64_t>(g_PushConstants.SharedConstants + 528 + (i)*8)
#define g_PulledInput(l)           vk::RawBufferLoad<uint2>(g_PushConstants.SharedConstants + 560 + (l)*8)
#endif
#else
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#ifdef REBLUE_RECOMP
// DXC takes the PointSize builtin on a struct member only, not on a bare
// entry point parameter.
struct PointSizeOutput
{
    [[vk::builtin("PointSize")]] float size : PSIZE;
};
#endif

#define g_SpecConstants() g_SpecConstants

#else

#ifdef REBLUE_RECOMP
#define DEFINE_SHARED_CONSTANTS() \
    uint4 g_BooleansArr[2] : packoffset(c16); \
    uint g_SwappedTexcoords : packoffset(c18.x); \
    float2 g_HalfPixelOffset : packoffset(c18.y); \
    float g_AlphaThreshold : packoffset(c18.w); \
    uint g_SwappedNormals : packoffset(c19.x); \
    uint g_SwappedBinormals : packoffset(c19.y); \
    uint g_SwappedTangents : packoffset(c19.z); \
    uint g_SwappedBlendWeights : packoffset(c19.w); \
    uint g_SwappedPositions : packoffset(c20.x); \
    uint g_SintTexcoords : packoffset(c20.y); \
    float g_ShadowPcfScale : packoffset(c20.z); \
    uint g_AlphaTestFunc : packoffset(c20.w); \
    uint g_ClipPlaneEnable : packoffset(c21.z);     uint g_Uhend3nTexcoords : packoffset(c21.w); \
    float4 g_ClipPlanesArr[6] : packoffset(c22); \
    float2 g_NdcScale : packoffset(c28.x); \
    float2 g_NdcOffset : packoffset(c28.z); \
    float2 g_EdgeSnap : packoffset(c29.x);     uint g_Hend3IntegerTexcoords : packoffset(c29.z);     uint g_Hend3SignedTexcoords : packoffset(c29.w); \
    GOTHAM_SHARED_CONSTANTS()

#ifdef GOTHAM_RECOMP
// SV_VertexID leaves out the draw's base vertex on D3D12, so the host passes
// it (Xenos r0.x includes it).
#define GOTHAM_SHARED_CONSTANTS() \
    int g_VertexIndexBase : packoffset(c30.x); \
    uint g_UintTexcoords : packoffset(c30.y); \
    uint4 g_PulledStreamSizes : packoffset(c31); \
    uint4 g_PulledStreamStrides : packoffset(c32); \
    uint4 g_PulledInputsArr[PULL_MAX_LOCATIONS / 2] : packoffset(c35);
#define g_PulledInput(l) (((l) & 1) != 0 ? g_PulledInputsArr[(l) / 2].zw : g_PulledInputsArr[(l) / 2].xy)
#else
#define GOTHAM_SHARED_CONSTANTS()
#endif

#define g_ClipPlanes(i) (g_ClipPlanesArr[(i)])

#define g_Booleans(i) (g_BooleansArr[(i) / 4][(i) % 4])
#else
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);
#endif

uint g_SpecConstants();

#endif

#ifdef REBLUE_RECOMP
// Test Xenos boolean register N in the unified VS(0..127)/PS(128..255) file.
#define BOOL_BIT(n) ((g_Booleans((n) / 32u) & (1u << ((n) & 31u))) != 0)
#endif

#ifdef REBLUE_RECOMP
// Fixed-function alpha test: RB_COLORCONTROL ALPHA_FUNC against RB_ALPHA_REF.
// 0 never, 1 less, 2 equal, 3 less-equal, 4 greater, 5 not-equal,
// 6 greater-equal, 7 always. ALPHA_TEST is a macro, since the shared constants
// it reads are declared after this header.
void alphaTestCompare(float alpha, float ref, uint func)
{
    bool pass;
    switch (func)
    {
    case 0: pass = false; break;
    case 1: pass = alpha < ref; break;
    case 2: pass = alpha == ref; break;
    case 3: pass = alpha <= ref; break;
    case 4: pass = alpha > ref; break;
    case 5: pass = alpha != ref; break;
    case 6: pass = alpha >= ref; break;
    default: pass = true; break;
    }
    clip(pass ? 1.0 : -1.0);
}
#define ALPHA_TEST(alpha) alphaTestCompare(alpha, g_AlphaThreshold, g_AlphaTestFunc)

// User clip planes (PA_CL_CLIP_CNTL bits 0-5, planes in clip space), written
// at every VS exit from the clip-space position. A disabled plane keeps all.
#define USER_CLIP_DISTANCE(i, pos) \
    ((((g_ClipPlaneEnable >> (i)) & 1u) != 0) ? dot((pos), g_ClipPlanes(i)) : 1.0)
#define WRITE_USER_CLIP_DISTANCES(pos) \
    oClipDistance0 = float4(USER_CLIP_DISTANCE(0, pos), USER_CLIP_DISTANCE(1, pos), \
                            USER_CLIP_DISTANCE(2, pos), USER_CLIP_DISTANCE(3, pos)); \
    oClipDistance1 = float2(USER_CLIP_DISTANCE(4, pos), USER_CLIP_DISTANCE(5, pos))
// Resolution scaling: a vertex less than half a guest pixel inside a viewport
// edge moves onto the edge. The guest tests coverage at guest pixel centres,
// so such an edge covers the whole outer guest pixel, but at a host scale the
// sub-pixels nearer the edge would be missed. g_EdgeSnap is that half pixel
// in NDC, or zero at 1x.
#define SNAP_TO_VIEWPORT_EDGES(pos) \
    if (g_EdgeSnap.x > 0.0 && (pos).w > 0.0) { \
        float2 snapNdc = (pos).xy / (pos).w; \
        float2 snapInset = 1.0 - abs(snapNdc); \
        snapNdc = select(and(snapInset > 0.0, snapInset < g_EdgeSnap), sign(snapNdc), snapNdc); \
        (pos).xy = snapNdc * (pos).w; \
    }
#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

// SPEC_CONSTANT_CLAMP_7E3: NaN to 0, colour into 7e3's [0, 31.875], alpha
// (two bits) into [0, 1].
float4 clamp7e3(float4 c)
{
    c = select(isnan(c), 0.0, c);
    return float4(clamp(c.rgb, 0.0, 31.875), saturate(c.a));
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

// Vertex shader fetches: no derivatives, so an explicit level.
float4 tfetch2DLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float lod)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture), lod);
}

// Pixel shader fetches with both the computed and the register LOD: the
// register LOD biases the computed one (Xenia folds it into the gradients),
// on top of the sampler's (fetch constant's) bias.
float4 tfetch2DBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float bias)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture), bias);
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

#ifdef REBLUE_RECOMP
// Bilinear-filtered shadow compare over the four neighboring depth texels.
float shadowCmp2D(uint resourceDescriptorIndex, float2 texCoord, float ref)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    int2 dimensions = int2(getTexture2DDimensions(texture));
    float2 coord = texCoord * dimensions - 0.5;
    float2 weights = frac(coord);
    int2 base = int2(floor(coord));
    int2 c0 = clamp(base, int2(0, 0), dimensions - 1);
    int2 c1 = clamp(base + 1, int2(0, 0), dimensions - 1);
    float4 taps = float4(
        texture.Load(int3(c0, 0)).x,
        texture.Load(int3(c1.x, c0.y, 0)).x,
        texture.Load(int3(c0.x, c1.y, 0)).x,
        texture.Load(int3(c1, 0)).x) > ref;
    return lerp(lerp(taps.x, taps.y, weights.x), lerp(taps.z, taps.w, weights.x), weights.y);
}
#endif

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
	#ifdef REBLUE_RECOMP
    // BD's cube-shadow PCF issues up to 9 cube ops per shader (bd_mirror_cs_ps /
    // bd_glass_cs_ps); overflowing this array drops the stored directions and
    // every tfetchCube reads OOB -> point-light shadows vanish.
    float3 cubeMapDirections[9];
    #else
    float3 cubeMapDirections[2];
    #endif
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

float4 tfetch3DBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float bias)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord, bias);
}

float4 tfetchCubeBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float bias)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], bias);
}

float4 tfetch3DLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float lod)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord, lod);
}

float4 tfetchCubeLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float lod)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], lod);
}

#ifdef REBLUE_RECOMP
// 10:10:10:2 signed normalised (DEC3N, or DEC4N with w): x/511 per component,
// -512 clamped to -1, and w as a 2-bit signed integer.
float4 decodeDec3N(uint v)
{
    int4 c = int4(int(v << 22) >> 22, int(v << 12) >> 22, int(v << 2) >> 22, int(v) >> 30);
    return float4(max(float3(c.xyz) / 511.0, -1.0), float(c.w));
}

// Packed tangent-basis decode; IA binds as R32_UINT so lane .x carries the
// raw bits (asuint recovers them). SPEC_CONSTANT_R11G11B10_NORMAL is the
// 11:11:10 packing reblue and rr6 met, SPEC_CONSTANT_DEC3N_NORMAL PGR3's
// 10:10:10:2.
float4 tfetchR11G11B10(float4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_DEC3N_NORMAL)
        return decodeDec3N(asuint(value.x));
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        uint v = asuint(value.x);
        return float4(
            (v & 0x00000400 ? -1.0 : 0.0) + ((v & 0x3FF) / 1024.0),
            (v & 0x00200000 ? -1.0 : 0.0) + (((v >> 11) & 0x3FF) / 1024.0),
            (v & 0x80000000 ? -1.0 : 0.0) + (((v >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    return value;
}

// Undo the engine bswap32 16-bit-pair swap (.yxwz) for any 16-bit-packed semantic flagged in the mask.
float4 swapFloats(uint swappedMask, float4 value, uint semanticIndex)
{
    return (swappedMask & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}

#ifdef GOTHAM_RECOMP
// The integer form, for BLENDINDICES (a uint4 input, or a pulled fetch's bits):
// PGR3's minimap packs a colour index and an edge coordinate as SHORT2 there.
// Its swap mask is bits 16 + index of g_SwappedBlendWeights.
uint4 swapUints(uint swappedMask, uint4 value, uint semanticIndex)
{
    return (swappedMask & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}
#endif

// 11:11:10 TEXCOORDs (k_10_11_11: x in bits 0-10, y in 11-21, z in 22-31;
// PGR3), bound as one raw 32-bit lane. 'mask' marks the TEXCOORDs to decode,
// 'signedMask' the signed ones and 'integerMask' those fetched as integers
// rather than normalised (the UHEND3N/HEND3N/UHEND3/HEND3 family); the
// callers pass g_Uhend3nTexcoords, g_Hend3SignedTexcoords and
// g_Hend3IntegerTexcoords.
float4 unpackUhend3n(uint mask, uint signedMask, uint integerMask, float4 value, uint semanticIndex)
{
    uint bit = 1u << semanticIndex;
    if ((mask & bit) == 0)
        return value;
    uint v = asuint(value.x);
    float3 c;
    float3 scale;
    if ((signedMask & bit) != 0)
    {
        c = float3(int(v << 21) >> 21, int(v << 10) >> 21, int(v) >> 22);
        scale = float3(1023.0, 1023.0, 511.0);
    }
    else
    {
        c = float3(v & 0x7FF, (v >> 11) & 0x7FF, v >> 22);
        scale = float3(2047.0, 2047.0, 1023.0);
    }
    if ((integerMask & bit) == 0)
        c = max(c / scale, -1.0);
    return float4(c, 1.0);
}

// Recover X360 integer-cast-to-float TEXCOORDs from R16G16(B16A16)_UINT bindings (sign-extend low 16 bits).
float4 sintTexcoord(uint mask, float4 value, uint semanticIndex)
{
    if ((mask & (1u << semanticIndex)) != 0)
    {
        int4 si = (int4(asuint(value)) << 16) >> 16;
        return float4(si);
    }
    return value;
}

#ifdef GOTHAM_RECOMP
// Unsigned integer TEXCOORDs (UBYTE4, UINT1), bound UINT: Xenos converts the
// integer to float, where the input assembler hands over its bits.
float4 uintTexcoord(uint mask, float4 value, uint semanticIndex)
{
    return (mask & (1u << semanticIndex)) != 0 ? float4(asuint(value)) : value;
}

// The pulled streams (see PULL_FORMAT_*): raw SRVs on D3D12, device addresses
// on Vulkan. The host's guest buffer mirrors are already byte-swapped per
// 32 bits, as the input assembler reads them.
#ifdef __spirv__
uint pulledLoad(uint stream, uint offset)
{
    return vk::RawBufferLoad<uint>(g_PulledStreamAddress(stream) + offset);
}
#else
ByteAddressBuffer g_PulledStream0 : register(t0, space6);
ByteAddressBuffer g_PulledStream1 : register(t1, space6);
ByteAddressBuffer g_PulledStream2 : register(t2, space6);
ByteAddressBuffer g_PulledStream3 : register(t3, space6);

uint pulledLoad(uint stream, uint offset)
{
    switch (stream)
    {
    case 0: return g_PulledStream0.Load(offset);
    case 1: return g_PulledStream1.Load(offset);
    case 2: return g_PulledStream2.Load(offset);
    default: return g_PulledStream3.Load(offset);
    }
}
#endif

uint pulledFormatDwords(uint format)
{
    switch (format)
    {
    case PULL_FORMAT_R32G32_FLOAT:
    case PULL_FORMAT_R16G16B16A16_SNORM:
    case PULL_FORMAT_R16G16B16A16_UNORM:
    case PULL_FORMAT_R16G16B16A16_SINT:
    case PULL_FORMAT_R16G16B16A16_UINT:
    case PULL_FORMAT_R16G16B16A16_FLOAT:
        return 2;
    case PULL_FORMAT_R32G32B32_FLOAT:
    case PULL_FORMAT_R32G32B32_UINT:
        return 3;
    case PULL_FORMAT_R32G32B32A32_FLOAT:
        return 4;
    default:
        return 1;
    }
}

// One input of a pulled vertex shader: what the input assembler would deliver
// for 'desc' (g_PulledInput) at vertex 'index'. Integer formats come back as
// their bits, as an integer binding against the float4 input does. Reads past
// the stream's end return zero data, as the input assembler does.
float4 pullVertex(uint2 desc, uint4 sizes, uint4 strides, float index)
{
    uint stream = desc.x & 0xFF;
    uint format = desc.x >> 8;
    uint4 raw = 0;
    if (stream < PULL_MAX_STREAMS && format != PULL_FORMAT_NONE)
    {
        uint dwords = pulledFormatDwords(format);
        uint address = uint(max(index, 0.0)) * strides[stream] + desc.y;
        if (address + dwords * 4 <= sizes[stream])
        {
            raw.x = pulledLoad(stream, address);
            if (dwords > 1) raw.y = pulledLoad(stream, address + 4);
            if (dwords > 2) raw.z = pulledLoad(stream, address + 8);
            if (dwords > 3) raw.w = pulledLoad(stream, address + 12);
        }
    }

    uint4 bytes = uint4(raw.x, raw.x >> 8, raw.x >> 16, raw.x >> 24) & 0xFF;
    int4 shorts = int4(raw.x << 16, raw.x, raw.y << 16, raw.y) >> 16;
    uint4 ushorts = uint4(raw.x, raw.x >> 16, raw.y, raw.y >> 16) & 0xFFFF;
    switch (format)
    {
    case PULL_FORMAT_R32_FLOAT: return float4(asfloat(raw.x), 0.0, 0.0, 1.0);
    case PULL_FORMAT_R32G32_FLOAT: return float4(asfloat(raw.xy), 0.0, 1.0);
    case PULL_FORMAT_R32G32B32_FLOAT: return float4(asfloat(raw.xyz), 1.0);
    case PULL_FORMAT_R32G32B32A32_FLOAT: return asfloat(raw);
    case PULL_FORMAT_R32_UINT: return asfloat(uint4(raw.x, 0, 0, 1));
    case PULL_FORMAT_R32G32B32_UINT: return asfloat(uint4(raw.xyz, 1));
    case PULL_FORMAT_B8G8R8A8_UNORM: return float4(bytes.zyxw) / 255.0;
    case PULL_FORMAT_R8G8B8A8_UNORM: return float4(bytes) / 255.0;
    case PULL_FORMAT_R8G8B8A8_UINT: return asfloat(bytes);
    case PULL_FORMAT_R16G16_SNORM: return float4(max(float2(shorts.xy) / 32767.0, -1.0), 0.0, 1.0);
    case PULL_FORMAT_R16G16B16A16_SNORM: return max(float4(shorts) / 32767.0, -1.0);
    case PULL_FORMAT_R16G16_UNORM: return float4(float2(ushorts.xy) / 65535.0, 0.0, 1.0);
    case PULL_FORMAT_R16G16B16A16_UNORM: return float4(ushorts) / 65535.0;
    case PULL_FORMAT_R16G16_SINT: return asfloat(int4(shorts.xy, 0, 1));
    case PULL_FORMAT_R16G16B16A16_SINT: return asfloat(shorts);
    case PULL_FORMAT_R16G16_UINT: return asfloat(uint4(ushorts.xy, 0, 1));
    case PULL_FORMAT_R16G16B16A16_UINT: return asfloat(ushorts);
    case PULL_FORMAT_R16G16_FLOAT: return float4(f16tof32(ushorts.xy), 0.0, 1.0);
    case PULL_FORMAT_R16G16B16A16_FLOAT: return f16tof32(ushorts);
    default: return float4(0.0, 0.0, 0.0, 1.0);
    }
}
#endif
#else
float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}
#endif

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
