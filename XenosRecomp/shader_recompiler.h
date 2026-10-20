#pragma once

#include "shader.h"
#include "shader_code.h"

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

struct ShaderRecompiler : StringBuffer
{
    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElement> vertexElements;
    // (usage << 4 | usageIndex) of each declared vertex input, and the fetch
    // parameters its first vfetch used.
    std::unordered_set<uint32_t> declaredVertexInputs;
    std::unordered_map<uint32_t, uint64_t> vertexInputFetches;
#ifdef GOTHAM_RECOMP
    // Set by a vfetch whose index is not r0.x: the shader then pulls every
    // input from the raw streams (shader_common.h pullVertex). The VFETCH
    // macro defined at vfetchDefinePos picks pulled or input-assembler reads.
    bool pullsVertices = false;
    size_t vfetchDefinePos = 0;
    // A mini fetch reuses the address of the full fetch before it.
    uint32_t fullFetchSrcRegister = 0;
    uint32_t fullFetchSrcSwizzle = 0;
    bool fullFetchIndexRounded = false;
#endif
    // SetTextureLod's register, declared at textureLodDeclPos when used.
    bool usesTextureLod = false;
    size_t textureLodDeclPos = 0;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> samplers;
    std::unordered_map<uint32_t, uint32_t> ifEndLabels;
    // Structured if/else emission: instruction indices where a "} else {" is
    // emitted (then-branch close is part of the event, not ifEndLabels).
    std::unordered_set<uint32_t> elseLabels;
    uint32_t specConstantsMask = 0;
    // Names the snapshots taken for co-issued scalar ops (see recompile(AluInstruction)).
    uint32_t scalarSnapshotCount = 0;

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

#ifdef REBLUE_RECOMP
    bool hasShadowTexture = false;
    uint32_t shadowTapUVCount = 0;
    std::unordered_map<uint32_t, uint32_t> shadowTapSlots;
#endif

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    void recompile(const uint8_t* shaderData, const std::string_view& include);
};
