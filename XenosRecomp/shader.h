#pragma once

#include "constant_table.h"

struct Float4Definition
{
    be<uint16_t> registerIndex;
    be<uint16_t> count;
    be<uint32_t> physicalOffset;
};

struct Int4Definition
{
    be<uint16_t> registerIndex;
    be<uint16_t> count;
    be<uint32_t> values[];
};

struct DefinitionTable
{
    be<uint32_t> field0;
    be<uint32_t> field4;
    be<uint32_t> field8;
    be<uint32_t> fieldC;
    be<uint32_t> size;
    be<uint32_t> definitions[]; // float4, int4 and bool separated by null terminators
};

struct Shader
{
    be<uint32_t> physicalOffset;
    be<uint32_t> size;
    be<uint32_t> field8;
    be<uint32_t> fieldC; // svPos register: (fieldC >> 8) & 0xFF
    be<uint32_t> field10;
    be<uint32_t> interpolatorInfo; // interpolator count: (interpolatorInfo >> 5) & 0x1F
};

enum class DeclUsage : uint32_t
{
    Position = 0,
    BlendWeight = 1,
    BlendIndices = 2,
    Normal = 3,
    PointSize = 4,
    TexCoord = 5,
    Tangent = 6,
    Binormal = 7,
    TessFactor = 8,
    PositionT = 9,
    Color = 10,
    Fog = 11,
    Depth = 12,
    Sample = 13
};

struct VertexElement
{
    uint32_t address : 12;
    DeclUsage usage : 4;
    uint32_t usageIndex : 4;
};

struct Interpolator
{
    uint32_t usageIndex : 4;
    DeclUsage usage : 4;
    uint32_t reg : 4;
    uint32_t : 20;
};

struct VertexShader : Shader
{
    be<uint32_t> field18;
    be<uint32_t> vertexElementCount;
    be<uint32_t> field20;
    be<uint32_t> vertexElementsAndInterpolators[]; // field18 + vertex elements + interpolators
};

enum PixelShaderOutputs : uint32_t
{
    PIXEL_SHADER_OUTPUT_COLOR0 = 0x1,
    PIXEL_SHADER_OUTPUT_COLOR1 = 0x2,
    PIXEL_SHADER_OUTPUT_COLOR2 = 0x4,
    PIXEL_SHADER_OUTPUT_COLOR3 = 0x8,
    PIXEL_SHADER_OUTPUT_DEPTH = 0x10
};

struct PixelShader : Shader
{
    be<uint32_t> field18;
    be<PixelShaderOutputs> outputs;
    be<uint32_t> interpolators[];
};

struct ShaderContainer
{
    be<uint32_t> flags;
    be<uint32_t> virtualSize;
    be<uint32_t> physicalSize;
    be<uint32_t> fieldC;
    be<uint32_t> constantTableOffset;
    be<uint32_t> definitionTableOffset;
    be<uint32_t> shaderOffset;
    be<uint32_t> field1C;
    be<uint32_t> field20;
};

// Container revision 0x0E, emitted by the XDK 2.0.2135 shader compiler (Ridge
// Racer 6). The header stops at shaderOffset, float definitions carry their
// values inline instead of in the physical section, and the program always
// starts at the top of the physical section.
constexpr uint32_t SHADER_CONTAINER_REVISION_0E = 0x0E;
constexpr uint32_t SHADER_CONTAINER_REVISION_11 = 0x11;

inline uint32_t getShaderContainerRevision(uint32_t flags)
{
    return (flags >> 8) & 0xFF;
}

struct ShaderContainer0E
{
    be<uint32_t> flags;
    be<uint32_t> virtualSize;
    be<uint32_t> physicalSize;
    be<uint32_t> definitionTableOffset;
    be<uint32_t> constantTableOffset;
    be<uint32_t> shaderOffset;
};

struct DefinitionTable0E
{
    be<uint32_t> field0;
    be<uint32_t> field4;
    be<uint32_t> field8;
    be<uint32_t> fieldC;
    be<uint32_t> size; // bytes of definitions[]
    be<uint32_t> field14; // uninitialized padding
    be<uint32_t> definitions[]; // float4 then int4 runs, each null terminated
};

struct Float4Definition0E
{
    be<uint16_t> registerAddress; // see getFloat4DefinitionRegister0E
    be<uint16_t> count; // in floats
    be<uint32_t> values[];
};

// The compiler packs literals downward from c255, and the address advances by
// 0x10 per register. VS and PS addresses differ only in the top nibble.
inline uint32_t getFloat4DefinitionRegister0E(uint32_t registerAddress)
{
    return ((registerAddress & 0xFFF) >> 4) + 208;
}

struct Shader0E
{
    be<uint32_t> programControl;
    be<uint32_t> field4;
    be<uint32_t> field8;
    be<uint32_t> fieldC;
    be<uint32_t> interpolatorInfo; // interpolator count: (interpolatorInfo >> 5) & 0x1F
    be<uint32_t> field14;
};

struct VertexShader0E : Shader0E
{
    be<uint32_t> field18;
    be<uint32_t> vertexElementCount;
    be<uint32_t> field20; // trailing entry count after the interpolators
    be<uint32_t> field24;
    be<uint32_t> vertexElementsAndInterpolators[]; // field18 + vertex elements + interpolators + field20
};

struct PixelShader0E : Shader0E
{
    be<uint32_t> field18;
    be<PixelShaderOutputs> outputs;
    be<uint32_t> interpolators[];
};