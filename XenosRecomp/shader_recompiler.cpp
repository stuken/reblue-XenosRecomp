#include "shader_recompiler.h"
#include "shader_common.h"

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
#ifdef REBLUE_RECOMP
    // BD IA-decodes SNORM normals to float; swapFloats() undoes the engine int16-pair swap, DEC3N goes through tfetchR11G11B10().
    "float4", // NORMAL
#else
    "uint4", // NORMAL
#endif
    "float4", // PSIZE
    "float4", // TEXCOORD
#ifdef REBLUE_RECOMP
    "float4", // TANGENT
    "float4", // BINORMAL
#else
    "uint4", // TANGENT
    "uint4", // BINORMAL
#endif
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
#ifdef REBLUE_RECOMP
    #define REBLUE_VERTEX_LOCATION_ROW(usage, index, location) { DeclUsage::usage, index, location },
    REBLUE_VERTEX_INPUT_LOCATIONS(REBLUE_VERTEX_LOCATION_ROW)
    #undef REBLUE_VERTEX_LOCATION_ROW
#else
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
#endif
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 },
#ifdef PGR3_RECOMP
    // One PGR3 pair (vs_820A6980 -> ps_820A66D0) passes NORMAL0. Last, so the
    // other interpolators keep their registers.
    { DeclUsage::Normal, 0 },
#elif defined(PGR4_RECOMP)
    // PGR4 pairs pass NORMAL0 (24 vertex, 43 pixel shaders) and TANGENT0/1 (6
    // vertex, 22 pixel shaders). Last, so the other interpolators keep their
    // registers.
    { DeclUsage::Normal, 0 },
    { DeclUsage::Tangent, 0 },
    { DeclUsage::Tangent, 1 },
#endif
};

#ifdef PGR4_RECOMP
// Where SharedConstants keeps the vertex samplers' descriptor indices (bytes
// 736-799, c46-c49): the host's SharedConstants::vertexTexture2DIndices.
static constexpr uint32_t VERTEX_SAMPLER_INDICES_OFFSET = 736;
// Each sampler slot's fetch constant LOD bias, as a float: pixel samplers 0-15
// at bytes 800-863 (c50-c53), vertex samplers 0-3 at 864-879 (c54). The host's
// SharedConstants::samplerLodBiases and vertexSamplerLodBiases.
static constexpr uint32_t SAMPLER_LOD_BIASES_OFFSET = 800;
static constexpr uint32_t VERTEX_SAMPLER_LOD_BIASES_OFFSET = 864;
// The pixel samplers' variant descriptor indices (SAMPLER_VARIANT_*), 16 per
// variant from SAMPLER_VARIANT_NO_ANISO: bytes 880-1135 (c55-c70), the host's
// SharedConstants::samplerVariantIndices.
static constexpr uint32_t SAMPLER_VARIANT_INDICES_OFFSET = 880;

// Xenos' instruction filter fields: 3 (mag, min, mip) and 7 (anisotropy)
// defer to the fetch constant.
static constexpr uint32_t FILTER_USE_FETCH_CONST = 3;
static constexpr uint32_t ANISO_USE_FETCH_CONST = 7;

// The sampler variant a fetch's filter overrides select, or -1 for a
// combination SAMPLER_VARIANT_* does not cover.
static int samplerVariant(const TextureFetchInstruction& instr)
{
    const bool fcMagMin = instr.magFilter == FILTER_USE_FETCH_CONST && instr.minFilter == FILTER_USE_FETCH_CONST;
    const bool fcMip = instr.mipFilter == FILTER_USE_FETCH_CONST;
    const bool fcAniso = instr.anisoFilter == ANISO_USE_FETCH_CONST;
    const bool pointMagMin = instr.magFilter == 0 && instr.minFilter == 0;
    const bool linearMagMin = instr.magFilter == 1 && instr.minFilter == 1;
    if (fcMagMin && fcMip && fcAniso)
        return SAMPLER_VARIANT_NONE;
    if (fcMagMin && fcMip && instr.anisoFilter == 0)
        return SAMPLER_VARIANT_NO_ANISO;
    // Point filtering leaves anisotropy nothing to do.
    if (pointMagMin && fcMip)
        return SAMPLER_VARIANT_POINT;
    if (pointMagMin && instr.mipFilter == 0)
        return SAMPLER_VARIANT_POINT_MIP_POINT;
    if (linearMagMin && fcMip && fcAniso)
        return SAMPLER_VARIANT_LINEAR;
    return -1;
}
#endif

static constexpr std::string_view TEXTURE_DIMENSIONS[] =
{
    "2D",
    "3D", 
    "Cube" 
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    auto findResult = vertexElements.find(address);
    assert(findResult != vertexElements.end());

    uint32_t srcRegister = instr.srcRegister;
    uint32_t srcSwizzle = instr.srcSwizzle & 3;
#ifdef GOTHAM_RECOMP
    bool indexRounded = instr.isIndexRounded;
    if (instr.isMiniFetch)
    {
        srcRegister = fullFetchSrcRegister;
        srcSwizzle = fullFetchSrcSwizzle;
        indexRounded = fullFetchIndexRounded;
    }
    else
    {
        fullFetchSrcRegister = srcRegister;
        fullFetchSrcSwizzle = srcSwizzle;
        fullFetchIndexRounded = indexRounded;
    }
    // Only r0.x holds the vertex index; any other index register means the
    // input assembler cannot model this shader's fetches.
    if (srcRegister != 0 || srcSwizzle != 0)
        pullsVertices = true;
    if (instr.srcRegisterAm)
        println("\n#error XenosRecomp: relative vfetch index register at 0x{:X}", address);
#endif

    // The fetch index register, for review: the input assembler can only model
    // fetches indexed by the vertex index.
    println("// vfetch {}{} index r{}.{}", USAGE_SEMANTICS[uint32_t(findResult->second.usage)],
        uint32_t(findResult->second.usageIndex), srcRegister, "xyzw"[srcSwizzle]);
    indent();

    // Inputs are bound by semantic, so every fetch of one semantic has to read
    // the same stream data.
    {
        const uint64_t fetch = uint64_t(instr.constIndex) | (uint64_t(instr.constIndexSelect) << 5) |
            (uint64_t(instr.format) << 7) | (uint64_t(instr.numFormatAll) << 13) |
            (uint64_t(instr.signedRfModeAll) << 14) | (uint64_t(instr.formatCompAll) << 15) |
            (uint64_t(instr.stride) << 16) | (uint64_t(uint32_t(instr.offset) & 0x7FFFFF) << 24) |
            (uint64_t(instr.isMiniFetch) << 47) | (uint64_t(uint32_t(instr.expAdjust) & 0x3F) << 48);
        const uint32_t input = (uint32_t(findResult->second.usage) << 4) | findResult->second.usageIndex;
        auto [it, first] = vertexInputFetches.emplace(input, fetch);
        if (!first && it->second != fetch)
            println("\n#error XenosRecomp: {}{} is fetched with different parameters at 0x{:X}",
                USAGE_SEMANTICS[uint32_t(findResult->second.usage)], uint32_t(findResult->second.usageIndex), address);
    }

#ifdef REBLUE_RECOMP
    // Wrap each 16-bit-packed semantic in swapFloats() (per-usage mask); TEXCOORD also runs sintTexcoord() for raw-int bindings.
    switch (findResult->second.usage)
    {
#ifdef GOTHAM_RECOMP
    // The packing comes per element from the host's masks, not the spec
    // constants (tfetchPackedBasis).
    case DeclUsage::Normal:
        print("tfetchPackedBasis(g_Dec3NBasis, g_R11G11B10Basis, swapFloats(g_SwappedNormals, ");
        break;
    case DeclUsage::Tangent:
        print("tfetchPackedBasis(g_Dec3NBasis, g_R11G11B10Basis, swapFloats(g_SwappedTangents, ");
        break;
    case DeclUsage::Binormal:
        print("tfetchPackedBasis(g_Dec3NBasis, g_R11G11B10Basis, swapFloats(g_SwappedBinormals, ");
        break;
#else
    case DeclUsage::Normal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedNormals, ");
        break;
    case DeclUsage::Tangent:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedTangents, ");
        break;
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedBinormals, ");
        break;
#endif
    case DeclUsage::BlendWeight:
        print("swapFloats(g_SwappedBlendWeights, ");
        break;
    case DeclUsage::TexCoord:
    #ifdef GOTHAM_RECOMP
        print("unpackUhend3n(g_Uhend3nTexcoords, g_Hend3SignedTexcoords, g_Hend3IntegerTexcoords, "
              "uintTexcoord(g_UintTexcoords, sintTexcoord(g_SintTexcoords, swapFloats(g_SwappedTexcoords, ");
    #else
        print("sintTexcoord(g_SintTexcoords, swapFloats(g_SwappedTexcoords, ");
    #endif
        break;
    case DeclUsage::Position:
    #ifdef GOTHAM_RECOMP
        // PGR3's SHORT4 positions are integers, not normalised: the host binds
        // them UINT and sets bit 16 + index of the TEXCOORD sign-extension mask.
        print("sintTexcoord(g_SintTexcoords, swapFloats(g_SwappedPositions, ");
    #else
        print("swapFloats(g_SwappedPositions, ");
    #endif
        break;
    #ifdef GOTHAM_RECOMP
    case DeclUsage::BlendIndices:
        // A uint4 input, converted to float as the input assembler hands it
        // over; a pulled fetch returns the integer's bits. 16-bit pairs get
        // the bswap32 swap first, as for the other semantics.
        print("float4(swapUints(g_SwappedBlendWeights, asuint(");
        break;
    #endif
    }
#else
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(");
        break;

    case DeclUsage::TexCoord:
        print("tfetchTexcoord(g_SwappedTexcoords, ");
        break;
    }
#endif

#ifdef GOTHAM_RECOMP
    {
        uint32_t location = ~0u;
        for (auto& usageLocation : USAGE_LOCATIONS)
        {
            if (usageLocation.usage == findResult->second.usage && usageLocation.usageIndex == findResult->second.usageIndex)
                location = usageLocation.location;
        }
        // The input-assembler input, its location and the Xenos fetch index
        // (rounded or floored, as the instruction says).
        print("VFETCH(i{}{}, {}, floor(r{}.{}{}))", USAGE_VARIABLES[uint32_t(findResult->second.usage)],
            uint32_t(findResult->second.usageIndex), location, srcRegister, "xyzw"[srcSwizzle],
            indexRounded ? " + 0.5" : "");
    }
#else
    print("i{}{}", USAGE_VARIABLES[uint32_t(findResult->second.usage)], uint32_t(findResult->second.usageIndex));
#endif

#ifdef REBLUE_RECOMP
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
    #ifdef GOTHAM_RECOMP
        print(", {}), {})", uint32_t(findResult->second.usageIndex),
              8 * (findResult->second.usage == DeclUsage::Normal    ? 0u
                   : findResult->second.usage == DeclUsage::Tangent ? 1u
                                                                    : 2u) +
                  uint32_t(findResult->second.usageIndex));
    #else
        print(", {}))", uint32_t(findResult->second.usageIndex));
    #endif
        break;

    case DeclUsage::TexCoord:
        print(", {}), {})", uint32_t(findResult->second.usageIndex),
              uint32_t(findResult->second.usageIndex));
    #ifdef GOTHAM_RECOMP
        print(", {}), {})", uint32_t(findResult->second.usageIndex),
              uint32_t(findResult->second.usageIndex));
    #endif
        break;

    case DeclUsage::BlendWeight:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    case DeclUsage::Position:
        print(", {})", uint32_t(findResult->second.usageIndex));
    #ifdef GOTHAM_RECOMP
        print(", {})", 16 + uint32_t(findResult->second.usageIndex));
    #endif
        break;
    #ifdef GOTHAM_RECOMP
    case DeclUsage::BlendIndices:
        print("), {}))", 16 + uint32_t(findResult->second.usageIndex));
        break;
    #endif
    }
#else
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        out += ')';
        break;

    case DeclUsage::TexCoord:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }
#endif

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    if (instr.opcode == FetchOpcode::SetTextureLod)
    {
        // The LOD that later fetches with useRegLod sample at.
        usesTextureLod = true;
        indent();
        if (instr.isPredicated)
            print("if ({}p0) ", instr.predCondition ? "" : "!");
        println("textureLod = r{}.{};", instr.srcRegister, SWIZZLES[instr.srcSwizzle & 3]);
        if (instr.srcRegisterAm)
            println("#error XenosRecomp: SetTextureLod with a relative source register");
        return;
    }

    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights)
        return;

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    // Xbox 360 D3D binds vertex sampler n to fetch constant 16 + n, while the
    // constant table names it by its sampler register n (PGR4's vertex
    // texture fetches read constants 16-19 for samplers 0-3).
    const uint32_t samplerRegister =
        (!isPixelShader && instr.constIndex >= 16) ? instr.constIndex - 16 : instr.constIndex;

    // Pixel samplers are host slots 0-15 and vertex samplers 16-19. A sampler
    // outside them marks every slot, so the host never skips one it reads.
    if (isPixelShader ? samplerRegister < 16 : samplerRegister < 4)
        samplerSlotMask |= 1u << (isPixelShader ? samplerRegister : 16 + samplerRegister);
    else
        samplerSlotMask |= 0xFFFFF;

    auto findResult = samplers.find(samplerRegister);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", samplerRegister);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        out += ");\n";
    }
#endif

#ifdef REBLUE_RECOMP
    // Stashed before the fetch statement: the kernel reuses UV registers as
    // fetch destinations (r6.y = tfetch2D(..., r6.yw)).
    const bool shadowTap = hasShadowTexture && !instr.isPredicated &&
        instr.opcode == FetchOpcode::TextureFetch &&
        instr.dimension == TextureDimension::Texture2D &&
        strcmp(constNamePtr, "ShadowTexture") == 0 &&
        instr.offsetX == 0 && instr.offsetY == 0 && shadowTapUVCount < 8;
    if (shadowTap)
    {
        indent();
        print("shadowTapUV[{}] = ", shadowTapUVCount);
        printSrcRegister(2);
        out += ";\n";
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        // Xenos lays a 1D texture out as a 2D one of height 1, so it is
        // fetched as 2D at v = 0.5 and the host has to bind it as a 2D
        // texture. (PGR3 uses this only for getWeights on a dummy sampler,
        // which needs just the width.)
        dimension = "2D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    // An explicit level, as Xenia does: the register LOD (SetTextureLod) when
    // the fetch asks for it, otherwise 0 in a vertex shader, which has no
    // derivatives; plus the instruction's LOD bias in 1/16ths.
    bool levelFetch = instr.opcode == FetchOpcode::TextureFetch &&
        (!isPixelShader || instr.useRegLod);
    // With the computed LOD as well, the register LOD is a bias on it, as
    // Xenia applies it: PGR4's road shaders fetch their base maps so, with the
    // register LOD at 0 and the fetch constant's bias doing the work.
    bool biasFetch = false;
#ifdef PGR4_RECOMP
    biasFetch = levelFetch && isPixelShader && instr.useCompLod &&
        instr.dimension != TextureDimension::Texture1D;
    if (biasFetch)
        levelFetch = false;
    // With neither, the level is explicit too: the instruction's (and the
    // fetch constant's) bias alone, as in Xenia. PGR4's metallic car paint
    // looks its colour up so in a 3x512 table whose sampler minifies with
    // point filtering, which a computed LOD above 0 selected.
    if (instr.opcode == FetchOpcode::TextureFetch && isPixelShader &&
        !instr.useCompLod && !instr.useRegLod)
        levelFetch = true;
    // The rest take the computed LOD, biased too: every fetch adds the fetch
    // constant's LOD bias (the sampler slot's _LodBias) in the shader, as
    // Xenia does, and the host's samplers carry none. D3D12 leaves a
    // sampler's MipLODBias off SampleLevel where Vulkan applies it, so only
    // this way do explicit levels get the bias, and on both backends alike.
    if (instr.opcode == FetchOpcode::TextureFetch && !levelFetch)
        biasFetch = true;
#endif
    if (levelFetch)
        out += "Level";
    if (biasFetch)
        out += "Bias";

    // A filter override reads its variant's sampler (pixel shaders; no PGR4
    // vertex fetch overrides the fetch constant).
    std::string samplerSuffix;
#ifdef PGR4_RECOMP
    if (instr.opcode == FetchOpcode::TextureFetch)
    {
        const int variant = samplerVariant(instr);
        if (variant < 0 || (variant != SAMPLER_VARIANT_NONE && !isPixelShader))
        {
            println("\n#error XenosRecomp: texture fetch filter override mag={} min={} mip={} aniso={} ({} shader) has no sampler variant",
                uint32_t(instr.magFilter), uint32_t(instr.minFilter), uint32_t(instr.mipFilter),
                uint32_t(instr.anisoFilter), isPixelShader ? "pixel" : "vertex");
        }
        else if (variant != SAMPLER_VARIANT_NONE)
        {
            samplerSuffix = std::to_string(variant);
        }
    }
#endif
    print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex{2}, ", constNamePtr, dimension, samplerSuffix);
    // Unnormalized coordinates address texels: scaled by the texture's size,
    // as Xenia does (not for cubes, whose coordinates are directions).
    bool denorm = false;
#ifdef PGR4_RECOMP
    denorm = instr.texCoordDenorm && instr.dimension != TextureDimension::TextureCube;
    if (denorm)
        print("denormTexCoord{0}({1}_Texture{0}DescriptorIndex, ", dimension, constNamePtr);
#endif
    if (instr.dimension == TextureDimension::Texture1D)
        out += "float2(";
    printSrcRegister(componentCount);
    if (instr.dimension == TextureDimension::Texture1D)
        out += ", 0.5)";
    if (denorm)
        out += ")";

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        print(", float2({}, 0)", instr.offsetX * 0.5f);
        break;
    case TextureDimension::Texture2D:
        print(", float2({}, {})", instr.offsetX * 0.5f, instr.offsetY * 0.5f);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    if (levelFetch || biasFetch)
    {
        usesTextureLod |= bool(instr.useRegLod);
        print(", {}{}", instr.useRegLod ? "textureLod + " : "", instr.lodBias / 16.0f);
#ifdef PGR4_RECOMP
        print(" + {}_LodBias", constNamePtr);
#endif
    }

    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

#ifdef REBLUE_RECOMP
    if (hasShadowTexture)
    {
        for (size_t i = 0; i < 4; i++)
        {
            if (getDestSwizzle(instr.dstSwizzle, i) <= FetchDestinationSwizzle::One)
                shadowTapSlots.erase(uint32_t(instr.dstRegister * 4 + i));
        }
        if (shadowTap)
        {
            for (size_t i = 0; i < 4; i++)
            {
                if (getDestSwizzle(instr.dstSwizzle, i) == FetchDestinationSwizzle::X)
                    shadowTapSlots[uint32_t(instr.dstRegister * 4 + i)] = shadowTapUVCount;
            }
            ++shadowTapUVCount;
        }
    }
#endif

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent(); 
        out += "{\n";
        ++indentation;
    }

    // Xenos co-issues an ALU instruction's vector and scalar halves, and both
    // read their sources before either writes. The vector result is emitted
    // first, so a scalar source naming the vector destination register is
    // read from a snapshot taken before the vector write.
    uint32_t scalarSnapshotRegister = UINT32_MAX;
    std::string scalarSnapshotName;

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                if (reg == scalarSnapshotRegister &&
                    (operand == SCALAR_0 || operand == SCALAR_1 || operand == SCALAR_CONSTANT_1))
                    regFormatted = scalarSnapshotName;
                else
                    regFormatted = fmt::format("r{}", reg);
            }
            else
            {
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        assert(!instr.const0Relative && !instr.const1Relative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!instr.const0Relative && !instr.const1Relative);
                    regFormatted = fmt::format("c{}", reg);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            case ExportRegister::VSPointSizeEdgeFlagKillVertex:
                // x point size, y edge flag, z kill vertex. Exported only by
                // point-sprite shaders (SQ_PROGRAM_CNTL VS export mode 2). The
                // host has no sprite expansion of its own, so the value is
                // kept in a local for the renderer to pick up.
                exportRegister = "oPts";
                break;

            default:
            {
                auto findResult = interpolators.find(instr.vectorDest);
                assert(findResult != interpolators.end());
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    if (vectorWriteMask != 0 && exportRegister.empty() && instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        // SCALAR_0/1 read src3 when it selects a register; the constant forms
        // (MULsc etc.) take their register operand from SCALAR_CONSTANT_1.
        const uint32_t constantFormRegister =
            (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
        if ((instr.src3Select && (instr.src3Register & 0x3F) == instr.vectorDest) ||
            constantFormRegister == instr.vectorDest)
        {
            scalarSnapshotRegister = instr.vectorDest;
            scalarSnapshotName = fmt::format("s{}", scalarSnapshotCount++);
            // Declared, then assigned: inside the switch (pc) state machine a
            // later case label may follow, and jumping past an initialised
            // declaration is an error.
            indent();
            println("float4 {}; {} = r{};", scalarSnapshotName, scalarSnapshotName, instr.vectorDest);
        }
    }

#ifdef REBLUE_RECOMP
    // A Sgt/Sge whose sources are ShadowTexture taps re-emits per lane as a
    // filtered compare from the stashed tap UV, turning the guest's binary PCF
    // into hardware-equivalent bilinear PCF. Anything else falls through.
    bool shadowCmpRewritten = false;
    if (vectorWriteMask != 0 && !instr.exportData && hasShadowTexture &&
        (instr.vectorOpcode == AluVectorOpcode::Sgt || instr.vectorOpcode == AluVectorOpcode::Sge) &&
        instr.src1Select && !instr.src1Negate && (instr.src1Register & 0x80) == 0 &&
        instr.src2Select && !instr.src2Negate && (instr.src2Register & 0x80) == 0)
    {
        const uint32_t srcA = instr.src1Register & 0x3F;
        const uint32_t srcB = instr.src2Register & 0x3F;
        bool allTaps = instr.vectorDest != srcA && instr.vectorDest != srcB;
        for (size_t i = 0; i < 4 && allTaps; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                allTaps = shadowTapSlots.find(srcA * 4 + (((instr.src1Swizzle >> (i * 2)) + i) & 0x3)) != shadowTapSlots.end();
        }
        if (allTaps)
        {
            for (size_t i = 0; i < 4; i++)
            {
                if (((vectorWriteMask >> i) & 0x1) == 0)
                    continue;
                const uint32_t compA = ((instr.src1Swizzle >> (i * 2)) + i) & 0x3;
                const uint32_t compB = ((instr.src2Swizzle >> (i * 2)) + i) & 0x3;
                indent();
                println("r{}.{} = shadowCmp2D(ShadowTexture_Texture2DDescriptorIndex, shadowTapUV[{}], r{}.{});",
                    instr.vectorDest, SWIZZLES[i], shadowTapSlots[srcA * 4 + compA], srcB, SWIZZLES[compB]);
            }
            shadowCmpRewritten = true;
        }
    }

    if (vectorWriteMask != 0 && !shadowCmpRewritten)
#else
    if (vectorWriteMask != 0)
#endif
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            print("max({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            print("min({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
        {
            // Xenos cube takes the direction from src1's swizzled lanes (z, w, x);
            // the canonical emit is src.zzxy which makes this src.xyz, but other
            // compilers emit e.g. src.yyzx (Blue Dragon), so honor the swizzle.
            const uint32_t reg = instr.src1Register & 0x3F;
            const bool srcAbs = (instr.src1Register & 0x80) != 0;
            auto lane = [&](uint32_t i) { return SWIZZLES[((instr.src1Swizzle >> (i * 2)) + i) & 0x3]; };
            print("cube({}{}r{}.{}{}{}{}{}, cubeMapData)",
                instr.src1Negate ? "-" : "", srcAbs ? "abs(" : "", reg,
                lane(2), lane(3), lane(0), lane(0), srcAbs ? ")" : "");
            break;
        }

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

#ifdef REBLUE_RECOMP
    if (hasShadowTexture && vectorWriteMask != 0 && exportRegister.empty())
    {
        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                shadowTapSlots.erase(uint32_t(instr.vectorDest * 4 + i));
        }
    }
#endif

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            print("max({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            print("min({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(rcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
#ifdef REBLUE_RECOMP
            // PRED_SETINV: src==1 -> 0, src==0 -> 1, else passthrough.
            print("{0} == 1.0 ? 0.0 : ({0} == 0.0 ? 1.0 : {0})", op(SCALAR_0));
#else
            print("{0} == 0.0 ? 1.0 : {0}", op(SCALAR_0));
#endif
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

#ifdef REBLUE_RECOMP
    if (hasShadowTexture && scalarWriteMask != 0 && exportRegister.empty())
    {
        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                shadowTapSlots.erase(uint32_t(instr.scalarDest * 4 + i));
        }
    }
#endif

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);
    const auto shaderContainer0E = reinterpret_cast<const ShaderContainer0E*>(shaderData);
    const bool isRevision0E = getShaderContainerRevision(shaderContainer->flags) == SHADER_CONTAINER_REVISION_0E;

    assert((shaderContainer->flags & 0xFFFF0000) == 0x102A0000);
    assert(isRevision0E || getShaderContainerRevision(shaderContainer->flags) == SHADER_CONTAINER_REVISION_11);

    const uint32_t constantTableOffset = isRevision0E ? shaderContainer0E->constantTableOffset : shaderContainer->constantTableOffset;
    const uint32_t definitionTableOffset = isRevision0E ? shaderContainer0E->definitionTableOffset : shaderContainer->definitionTableOffset;
    const uint32_t shaderOffset = isRevision0E ? shaderContainer0E->shaderOffset : shaderContainer->shaderOffset;
    assert(constantTableOffset != NULL);

    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

#if defined(REBLUE_RECOMP) && !defined(RR6_RECOMP) && !defined(GOTHAM_RECOMP)
        // Blue Dragon's sun-shadow PCF rewrite. RR6 also names a sampler
        // ShadowTexture, but its kernel is not BD's, so it stays untouched.
        if (isPixelShader && constantInfo->registerSet == RegisterSet::Sampler &&
            strcmp(constantName, "ShadowTexture") == 0)
        {
            hasShadowTexture = true;
        }
#endif

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                // A relative index addresses the whole register file: one
                // that leaves it reads 0 instead of memory either side.
                println("#define {0}(INDEX) select((INDEX) >= -{3} && (INDEX) < {1}, vk::RawBufferLoad<float4>(g_PushConstants.{2}ShaderConstants + ({3} + clamp(INDEX, -{3}, {4})) * 16, 0x10), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10)",
                    constantName, shaderName, constantInfo->registerIndex * 16);
            }
            
#ifdef REBLUE_RECOMP
            // BD aliases a singleton constant over an array slot; the wider registrant wins so body and cbuffer agree.
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            {
                uint32_t reg = constantInfo->registerIndex + j;
                auto it = float4Constants.find(reg);
                if (it == float4Constants.end() || it->second->registerCount < constantInfo->registerCount)
                    float4Constants[reg] = constantInfo;
            }
#else
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);
#endif

            break;
        }

        case RegisterSet::Sampler:
        {
#ifdef PGR4_RECOMP
            // Vertex samplers 0-3 (the device's sampler slots 16-19) have their
            // own descriptor indices after the shared block's other fields,
            // one float4 per dimension and one for the samplers.
            if (!isPixelShader)
            {
                for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
                {
                    println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                        constantName, TEXTURE_DIMENSIONS[j], VERTEX_SAMPLER_INDICES_OFFSET + j * 16 + constantInfo->registerIndex * 4);
                }

                println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, VERTEX_SAMPLER_INDICES_OFFSET + std::size(TEXTURE_DIMENSIONS) * 16 + constantInfo->registerIndex * 4);
                println("#define {}_LodBias vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + {})",
                    constantName, VERTEX_SAMPLER_LOD_BIASES_OFFSET + constantInfo->registerIndex * 4);

                samplers.emplace(constantInfo->registerIndex, constantName);
                break;
            }
#endif
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, TEXTURE_DIMENSIONS[j], j * 64 + constantInfo->registerIndex * 4);
            }

            println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                constantName, std::size(TEXTURE_DIMENSIONS) * 64 + constantInfo->registerIndex * 4);
#ifdef PGR4_RECOMP
            println("#define {}_LodBias vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + {})",
                constantName, SAMPLER_LOD_BIASES_OFFSET + constantInfo->registerIndex * 4);
            for (uint32_t v = 1; v < SAMPLER_VARIANT_COUNT; v++)
            {
                println("#define {}_SamplerDescriptorIndex{} vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, v, SAMPLER_VARIANT_INDICES_OFFSET + (v - 1) * 64 + constantInfo->registerIndex * 4);
            }
#endif

            samplers.emplace(constantInfo->registerIndex, constantName);
            break;
        }

        }
    }

#ifdef REBLUE_RECOMP
    // BD bodies reference unnamed sampler slots; emit fallback descriptor-index defines for any slot 0..15 the table didn't name.
    for (uint32_t r = 0; r < 16; r++)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("#define s{}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                r, TEXTURE_DIMENSIONS[j], j * 64 + r * 4);
        }
        println("#define s{}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
            r, std::size(TEXTURE_DIMENSIONS) * 64 + r * 4);
#ifdef PGR4_RECOMP
        println("#define s{}_LodBias vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + {})",
            r, SAMPLER_LOD_BIASES_OFFSET + r * 4);
        for (uint32_t v = 1; v < SAMPLER_VARIANT_COUNT; v++)
        {
            println("#define s{}_SamplerDescriptorIndex{} vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                r, v, SAMPLER_VARIANT_INDICES_OFFSET + (v - 1) * 64 + r * 4);
        }
#endif
    }
#endif

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
#ifdef REBLUE_RECOMP
            // Only the alias winner gets a packoffset slot; a loser would overlap it in the cbuffer and fail DXC.
            auto winner = float4Constants.find(constantInfo->registerIndex);
            if (winner == float4Constants.end() || winner->second != constantInfo)
                continue;
#endif

            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;
                println("#define {0}(INDEX) select((INDEX) >= -{3} && (INDEX) < {1}, {0}[clamp(INDEX, -{3}, {2})], 0.0)",
                    constantName, tailCount, tailCount - 1, constantInfo->registerIndex.get());
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

#ifdef PGR4_RECOMP
            if (!isPixelShader)
            {
                constexpr uint32_t base = VERTEX_SAMPLER_INDICES_OFFSET / 16;
                for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
                {
                    println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                        constantName, TEXTURE_DIMENSIONS[j], base + j, SWIZZLES[constantInfo->registerIndex % 4]);
                }

                println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                    constantName, base + std::size(TEXTURE_DIMENSIONS), SWIZZLES[constantInfo->registerIndex % 4]);
                println("\tfloat {}_LodBias : packoffset(c{}.{});",
                    constantName, VERTEX_SAMPLER_LOD_BIASES_OFFSET / 16, SWIZZLES[constantInfo->registerIndex % 4]);
                continue;
            }
#endif
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
#ifdef PGR4_RECOMP
            println("\tfloat {}_LodBias : packoffset(c{}.{});",
                constantName, SAMPLER_LOD_BIASES_OFFSET / 16 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            for (uint32_t v = 1; v < SAMPLER_VARIANT_COUNT; v++)
            {
                println("\tuint {}_SamplerDescriptorIndex{} : packoffset(c{}.{});",
                    constantName, v, SAMPLER_VARIANT_INDICES_OFFSET / 16 + (v - 1) * 4 + constantInfo->registerIndex / 4,
                    SWIZZLES[constantInfo->registerIndex % 4]);
            }
#endif
        }
    }

#ifdef REBLUE_RECOMP
    // Mirror the SPIR-V fallback: packoffset slots for any unnamed sampler index 0..15.
    for (uint32_t r = 0; r < 16; r++)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("\tuint s{}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                r, TEXTURE_DIMENSIONS[j], j * 4 + r / 4, SWIZZLES[r % 4]);
        }
        println("\tuint s{}_SamplerDescriptorIndex : packoffset(c{}.{});",
            r, 4 * std::size(TEXTURE_DIMENSIONS) + r / 4, SWIZZLES[r % 4]);
#ifdef PGR4_RECOMP
        println("\tfloat s{}_LodBias : packoffset(c{}.{});",
            r, SAMPLER_LOD_BIASES_OFFSET / 16 + r / 4, SWIZZLES[r % 4]);
        for (uint32_t v = 1; v < SAMPLER_VARIANT_COUNT; v++)
        {
            println("\tuint s{}_SamplerDescriptorIndex{} : packoffset(c{}.{});",
                r, v, SAMPLER_VARIANT_INDICES_OFFSET / 16 + (v - 1) * 4 + r / 4, SWIZZLES[r % 4]);
        }
#endif
    }
#endif

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
#ifdef REBLUE_RECOMP
            // Key named bools by the unified VS(0..127)/PS(128..255) bit so CF tests (also unified) resolve them.
            const uint32_t unifiedBit = uint32_t(constantInfo->registerIndex) + (isPixelShader ? 128u : 0u);
            println("\t#define {} BOOL_BIT({})", constantName, unifiedBit);
            boolConstants.emplace(unifiedBit, constantName);
#else
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            boolConstants.emplace(constantInfo->registerIndex, constantName);
#endif
        }
    }

    out += '\n';

    // Both container revisions reduced to the fields the emitter reads.
    struct
    {
        uint32_t codeOffset; // from the start of the physical section
        uint32_t codeSize;
        uint32_t svPosRegister; // 0xFF: none
        uint32_t interpolatorCount;
        const be<uint32_t>* interpolators; // PS inputs / VS outputs
        uint32_t vertexElementCount;
        const be<uint32_t>* vertexElements;
        uint32_t pixelOutputs;
    } shader{};

    if (isRevision0E)
    {
        const auto shader0E = reinterpret_cast<const Shader0E*>(shaderData + shaderOffset);
        shader.codeOffset = 0;
        shader.codeSize = shaderContainer0E->physicalSize;
        // No 0x0E header field locates a position input, and no RR6 pixel
        // shader reads a register that its interpolators or code leave unset.
        shader.svPosRegister = 0xFF;
        shader.interpolatorCount = (shader0E->interpolatorInfo >> 5) & 0x1F;
        if (isPixelShader)
        {
            const auto pixelShader = reinterpret_cast<const PixelShader0E*>(shader0E);
            shader.interpolators = pixelShader->interpolators;
            shader.pixelOutputs = pixelShader->outputs;
        }
        else
        {
            const auto vertexShader = reinterpret_cast<const VertexShader0E*>(shader0E);
            shader.vertexElementCount = vertexShader->vertexElementCount;
            shader.vertexElements = vertexShader->vertexElementsAndInterpolators + vertexShader->field18;
            shader.interpolators = shader.vertexElements + shader.vertexElementCount;
        }
    }
    else
    {
        const auto shader11 = reinterpret_cast<const Shader*>(shaderData + shaderOffset);
        shader.codeOffset = shader11->physicalOffset;
        shader.codeSize = shader11->size;
        shader.svPosRegister = (shader11->fieldC >> 8) & 0xFF;
        shader.interpolatorCount = (shader11->interpolatorInfo >> 5) & 0x1F;
        if (isPixelShader)
        {
            const auto pixelShader = reinterpret_cast<const PixelShader*>(shader11);
            shader.interpolators = pixelShader->interpolators;
            shader.pixelOutputs = pixelShader->outputs;
        }
        else
        {
            const auto vertexShader = reinterpret_cast<const VertexShader*>(shader11);
            shader.vertexElementCount = vertexShader->vertexElementCount;
            shader.vertexElements = vertexShader->vertexElementsAndInterpolators + vertexShader->field18;
            shader.interpolators = shader.vertexElements + shader.vertexElementCount;
        }
    }

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

#ifdef GOTHAM_RECOMP
    vfetchDefinePos = out.size();
#endif
    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
        {
#ifdef REBLUE_RECOMP
            const char* interpolation = (usage == DeclUsage::Color) ? "centroid " : "";
#else
            const char* interpolation = "";
#endif
            println("\tin {3}float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)], interpolation);
        }

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        if (shader.pixelOutputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (shader.pixelOutputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (shader.pixelOutputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (shader.pixelOutputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (shader.pixelOutputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
    }
    else
    {
        for (uint32_t i = 0; i < shader.vertexElementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            value = shader.vertexElements[i];

            // The table has one element per vfetch, so a semantic fetched in
            // more than one place (PGR3 has six such shaders) is listed more
            // than once. Declare its input once; every fetch address still
            // maps to it, and recompile(VertexFetchInstruction) checks that
            // the fetches agree.
            vertexElements.emplace(uint32_t(vertexElement.address), vertexElement);
            if (!declaredVertexInputs.insert((uint32_t(vertexElement.usage) << 4) | vertexElement.usageIndex).second)
                continue;

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

        #ifdef UNLEASHED_RECOMP
            if ((vertexElement.usage == DeclUsage::TexCoord && vertexElement.usageIndex == 2 && isMetaInstancer) ||
                (vertexElement.usage == DeclUsage::Position && vertexElement.usageIndex == 1))
            {
                usageType = "uint4";
            }
        #endif

            out += '\t';

            bool locatedInput = false;
            for (auto& usageLocation : USAGE_LOCATIONS)
            {
                if (usageLocation.usage == vertexElement.usage && usageLocation.usageIndex == vertexElement.usageIndex)
                {
                    print("[[vk::location({})]] ", usageLocation.location);
                    locatedInput = true;
                    break;
                }
            }
            if (!locatedInput)
            {
                // Without a location SPIR-V rejects the whole signature; say why.
                println("\n#error XenosRecomp: no vertex input location for {}{}; add it to the game's *_VERTEX_INPUT_LOCATIONS table",
                    USAGE_SEMANTICS[uint32_t(vertexElement.usage)], uint32_t(vertexElement.usageIndex));
                out += '\t';
            }

            println("in {0} i{1}{2} : {3}{2},", usageType, USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex), USAGE_SEMANTICS[uint32_t(vertexElement.usage)]);
        }

    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iVertexId : SV_VertexID,\n";
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #elif defined(GOTHAM_RECOMP)
        // Xenos loads the vertex index into r0.x.
        out += "\tin uint iVertexId : SV_VertexID,\n";
    #endif

        out += "\tout float4 oPos : SV_Position";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);

    #ifdef REBLUE_RECOMP
        // Last, so the interpolators keep the registers the pixel shader's
        // input signature expects.
        out += ",\n\tout float4 oClipDistance0 : SV_ClipDistance0";
        out += ",\n\tout float2 oClipDistance1 : SV_ClipDistance1";
        // Vulkan leaves the size of a point undefined unless the vertex shader
        // writes it (VUID-VkGraphicsPipelineCreateInfo-topology-08773). The
        // host draws guest points one pixel wide, as D3D12 does.
        out += "\n#ifdef __spirv__\n";
        out += "\t, out PointSizeOutput oPointSize\n";
        out += "#endif\n";
    #endif
    }

    out += ")\n";
    out += "{\n";

#ifdef REBLUE_RECOMP
    if (!isPixelShader)
        out += "#ifdef __spirv__\n\toPointSize.size = 1.0;\n#endif\n";
#endif

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";
    }
#endif

    // One literal float4 register, shared by both container revisions.
    auto printFloat4Definition = [&](uint32_t reg, const be<uint32_t>* value)
    {
#ifdef REBLUE_RECOMP
        // BD bakes the sun-shadow receiver PCF tap offsets as shader
        // literals in 1/1024-of-the-map UV units, folded into 0.5+k
        // and 1.0+k anchors plus the raw 1/1024 frac-stagger stride.
        // Their world footprint scales with the shadow coverage box
        // (bd_shadow_distance), so re-anchor each such component on
        // g_ShadowPcfScale (SharedConstants c20.z, byte 328, =
        // max(1/distance, 1024/dimension)). Bit-exact identity at
        // scale 1.0. Gated to pixel shaders sampling ShadowTexture;
        // the +/-3/1024 window (anchors excluded) matches exactly the
        // kernel literals across every BD receiver and nothing else.
        auto shadowKernelComponent = [&](uint32_t u) -> std::string
        {
            if (hasShadowTexture)
            {
                if (u == 0x3A800000)
                    return fmt::format("(asfloat(0x{:X}u) * g_ShadowPcfScale)", u);
                float f;
                memcpy(&f, &u, sizeof(f));
                for (float anchor : { 0.5f, 1.0f })
                {
                    float d = f > anchor ? f - anchor : anchor - f;
                    if (f != anchor && d <= 3.0f / 1024.0f)
                        return fmt::format("({} + (asfloat(0x{:X}u) - {}) * g_ShadowPcfScale)", anchor, u, anchor);
                }
            }
            return fmt::format("asfloat(0x{:X}u)", u);
        };
        println("\tfloat4 c{} = float4({}, {}, {}, {});",
            reg,
            shadowKernelComponent(value[0].get()), shadowKernelComponent(value[1].get()),
            shadowKernelComponent(value[2].get()), shadowKernelComponent(value[3].get()));
#else
        println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
            reg, value[0].get(), value[1].get(), value[2].get(), value[3].get());
#endif
    };

    if (isRevision0E && definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable0E*>(shaderData + definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition0E*>(definitions);
            const uint32_t baseRegister = getFloat4DefinitionRegister0E(definition->registerAddress);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
                printFloat4Definition(baseRegister + i, definition->values + i * 4);
            definitions += 1 + definition->count;
        }
        ++definitions;
        // No RR6 shader defines int4 literals, so their 0x0E encoding is unknown.
        if (*definitions != 0)
            out += "#error XenosRecomp: container revision 0x0E int4 definitions are not supported\n";
        out += "\n";
    }
    else if (definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                printFloat4Definition(definition->registerIndex + i - (isPixelShader ? 256 : 0), value);
                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, x, y, z, w);
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    for (uint32_t i = 0; i < shader.interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };

        value = shader.interpolators[i];

        if (isPixelShader)
        {
            println("\tfloat4 r{} = i{}{};", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }

    if (!isPixelShader)
    {
    #if defined(UNLEASHED_RECOMP)
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #elif defined(REBLUE_RECOMP)
        // Always define SV_Position so a skipped position-write block doesn't leave it undef.
        out += "\toPos = 0.0;\n";
        out += "\toClipDistance0 = 1.0;\n";
        out += "\toClipDistance1 = 1.0;\n";
    #endif

        // Point size / edge flag / kill vertex export (export register 63).
        out += "\tfloat4 oPts = 0.0;\n";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == shader.svPosRegister)
            {
                out += "float4((iPos.xy - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #elif defined(GOTHAM_RECOMP)
            else if (!isPixelShader && i == 0)
            {
                out += "float4(float(int(iVertexId) + g_VertexIndexBase), 0.0, 0.0, 0.0);\n";
            }
        #endif
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    out += "\tfloat ps = 0.0;\n";
    textureLodDeclPos = out.size();
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
#ifdef REBLUE_RECOMP
        if (hasShadowTexture)
            out += "\tfloat2 shadowTapUV[8] = (float2[8])0;\n";
#endif
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader.codeOffset);

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader.codeSize;
    bool simpleControlFlow = true;

    // Per-instruction control flow summary for the structurizer below.
    enum class CfKind : uint8_t { None, Cond, Uncond, LoopStart, LoopEnd };
    struct CfSummary
    {
        CfKind kind = CfKind::None;
        uint32_t target = 0;
    };
    std::vector<CfSummary> cfSummaries;
    std::vector<uint32_t> loopStarts;
    std::unordered_map<uint32_t, uint32_t> loopEndFor;
    uint32_t cfIndex = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                break;

            case ControlFlowOpcode::LoopStart:
                loopStarts.push_back(cfIndex);
                cfSummaries.push_back({ CfKind::LoopStart, 0 });
                break;

            case ControlFlowOpcode::LoopEnd:
                if (loopStarts.empty())
                    simpleControlFlow = false;
                else
                {
                    loopEndFor[loopStarts.back()] = cfIndex;
                    loopStarts.pop_back();
                }
                cfSummaries.push_back({ CfKind::LoopEnd, 0 });
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.direction || cfInstr.condJmp.address <= cfIndex)
                    simpleControlFlow = false;

                cfSummaries.push_back({
                    cfInstr.condJmp.isUnconditional ? CfKind::Uncond : CfKind::Cond,
                    uint32_t(cfInstr.condJmp.address) });
                break;
            }

            case ControlFlowOpcode::CondCall:
            case ControlFlowOpcode::Return:
                // Subroutines are not modeled; skipping them would silently
                // drop code, so fail the compile instead.
                println("#error XenosRecomp: control flow {} (call/return) is not supported",
                    cfInstr.opcode == ControlFlowOpcode::CondCall ? "cond_call" : "return");
                break;
            }

            if (cfSummaries.size() == cfIndex)
                cfSummaries.push_back({});

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);

            ++cfIndex;
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!loopStarts.empty())
        simpleControlFlow = false;

    if (simpleControlFlow)
    {
        // Recursive-descent structurizer. Xenos control flow in practice is
        // structured code lowered to forward jumps: "cjmp c, ELSE; then;
        // jmp MERGE; ELSE: else" diamonds where the compiler may retarget the
        // then-exit jump at an ENCLOSING merge point (jump chaining), plus
        // LoopStart/LoopEnd pairs. structure() walks a region [lo, hi) with
        // the continuation index "cont" (where control flows after hi): a
        // trailing unconditional jump to the continuation is a no-op after
        // structuring, a conditional jump opens an if (with an else branch
        // when the instruction before its target is an unconditional jump to
        // a legal merge point). Any other shape rejects the shader back to
        // the switch(pc) state machine.
        const uint32_t instrCount = (instrSize / 12) * 2;
        if (cfSummaries.size() > instrCount)
            cfSummaries.resize(instrCount);

        std::function<bool(uint32_t, uint32_t, uint32_t)> structure =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool
        {
            uint32_t i = lo;
            while (i < hi)
            {
                const CfSummary& summary = cfSummaries[i];
                switch (summary.kind)
                {
                case CfKind::Uncond:
                {
                    if (i + 1 == hi && (summary.target == cont || summary.target == hi))
                        i = hi;
                    else
                        return false;
                    break;
                }
                case CfKind::Cond:
                {
                    const uint32_t t = summary.target;
                    if (t > hi)
                    {
                        if (t != cont)
                            return false;
                        ++ifEndLabels[hi];
                        if (!structure(i + 1, hi, cont))
                            return false;
                        i = hi;
                    }
                    // A then-exit jump targeting the cond target itself is a
                    // no-op (empty else); fall through to the plain-if path,
                    // whose trailing-jump rule deletes it.
                    else if (t < hi && t - 1 > i && cfSummaries[t - 1].kind == CfKind::Uncond &&
                             cfSummaries[t - 1].target != t)
                    {
                        const uint32_t merge = cfSummaries[t - 1].target;
                        if (merge == cont || merge == hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[hi];
                            if (!structure(i + 1, t - 1, merge == cont ? cont : hi))
                                return false;
                            if (!structure(t, hi, cont))
                                return false;
                            i = hi;
                        }
                        else if (merge < hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[merge];
                            if (!structure(i + 1, t - 1, merge))
                                return false;
                            if (!structure(t, merge, merge))
                                return false;
                            i = merge;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        ++ifEndLabels[t];
                        if (!structure(i + 1, t, t == hi ? cont : t))
                            return false;
                        i = t;
                    }
                    break;
                }
                case CfKind::LoopStart:
                {
                    auto loopEnd = loopEndFor.find(i);
                    if (loopEnd == loopEndFor.end() || loopEnd->second >= hi)
                        return false;
                    if (!structure(i + 1, loopEnd->second, loopEnd->second))
                        return false;
                    i = loopEnd->second + 1;
                    break;
                }
                case CfKind::LoopEnd:
                    return false; // reached without its LoopStart

                default:
                    ++i;
                    break;
                }
            }
            return true;
        };

        simpleControlFlow = structure(0, instrCount, instrCount);
    }

    if (!simpleControlFlow)
    {
        ifEndLabels.clear();
        elseLabels.clear();
    }

    if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (!simpleControlFlow)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
                // Inner regions close first (validated nesting), then the
                // enclosing if's then-branch flips to its else-branch.
                if (elseLabels.count(pc) != 0)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                    indent();
                    out += "else\n";
                    indent();
                    out += "{\n";
                    ++indentation;
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
#ifdef REBLUE_RECOMP
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd ||
                                cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);
                // Gate the block on its boolean constant (Xenos cexec bN / cexec !bN); baseline ran it unconditionally.
                {
                    indent();
                    auto findResult = boolConstants.find(cfInstr.condExec.boolAddress);
                    if (findResult != boolConstants.end())
                        println("if ({}{})", cfInstr.condExec.condition ? "" : "!", findResult->second);
                    else
                        println("if ({}BOOL_BIT({}))", cfInstr.condExec.condition ? "" : "!", uint32_t(cfInstr.condExec.boolAddress));
                    indent();
                    out += "{\n";
                    ++indentation;
                    shouldCloseCurlyBracket = true;
                }
#else
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd || cfInstr.opcode == ControlFlowOpcode::CondExecEnd);
#endif
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                break;

            case ControlFlowOpcode::LoopStart:
                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else 
                {
                    out += "\t\t\taL = 0;\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
                if (simpleControlFlow)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional)
                {
                    // Structured mode: the then-branch exit jump was consumed by
                    // the "} else {" emitted at the next instruction.
                    if (!simpleControlFlow)
                    {
                        println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\tcontinue;\n";
                    }
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
#ifdef REBLUE_RECOMP
                        // Bools expand to BOOL_BIT(N) expressions; negate when the jump fires on a clear bit.
                        const bool jumpIfSet = cfInstr.condJmp.condition ^ simpleControlFlow;
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ({}{})", jumpIfSet ? "" : "!", findResult->second);
                        else
                            println("if ({}BOOL_BIT({}))", jumpIfSet ? "" : "!", uint32_t(cfInstr.condJmp.boolAddress));
#else
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else
                            println("if (b{} {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
#endif
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            auto instructionCode = code + address * 3;
            
            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            if (shouldReturn)
            {
                if (isPixelShader)
                {
#ifdef PGR4_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_CLAMP_7E3;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_CLAMP_7E3)\n";
                    indent();
                    out += "\toC0 = clamp7e3(oC0);\n";
#endif
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                #ifdef REBLUE_RECOMP
                    out += "\tALPHA_TEST(oC0.w);\n";
                #else
                    out += "\tclip(oC0.w - g_AlphaThreshold);\n";
                #endif

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                #endif
                    {
                    #ifdef REBLUE_RECOMP
                        out += "\tWRITE_USER_CLIP_DISTANCES(oPos);\n";
                        // g_NdcScale/g_NdcOffset map window-space positions
                        // to NDC while the guest viewport transform is off.
                        out += "\toPos.xy = oPos.xy * g_NdcScale + (g_NdcOffset + g_HalfPixelOffset) * oPos.w;\n";
                        out += "\tSNAP_TO_VIEWPORT_EDGES(oPos);\n";
                    #else
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    #endif
                    }
                }

                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!simpleControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }
    else
    {
        // Regions closing one past the last instruction never get visited by
        // the loop above; balance their braces here.
        auto findResult = ifEndLabels.find(pc);
        if (findResult != ifEndLabels.end())
        {
            for (uint32_t i = 0; i < findResult->second; i++)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }
    }

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";

    // Declared only where used, so other shaders' HLSL is unchanged. Inserted
    // before the VFETCH define, which sits earlier in the output.
    if (usesTextureLod)
        out.insert(textureLodDeclPos, "\tfloat textureLod = 0.0;\n");

#ifdef GOTHAM_RECOMP
    // Known only now that every fetch has been seen.
    out.insert(vfetchDefinePos, pullsVertices
        ? "// Pulls its vertices: a fetch index is computed, not the vertex index.\n"
          "#define VFETCH(input, location, index) pullVertex(g_PulledInput(location), g_PulledStreamSizes, g_PulledStreamStrides, index)\n\n"
        : "#define VFETCH(input, location, index) input\n\n");
#endif
}
