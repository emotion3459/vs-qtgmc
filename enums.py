from vstools import CustomIntEnum


class InputType(CustomIntEnum):
    INTERLACE = 0
    PROGRESSIVE = 1
    REPAIR = 2


class SearchPostProcess(CustomIntEnum):
    NONE = 0
    GAUSSBLUR = 1
    GAUSSBLUR_EDGESOFTEN = 2


class LosslessMode(CustomIntEnum):
    NONE = 0
    PRESHARPEN = 1
    POSTSMOOTH = 2


class NoiseDeintMode(CustomIntEnum):
    WEAVE = 0
    BOB = 1
    GENERATE = 2


class NoiseProcessMode(CustomIntEnum):
    NONE = 0
    DENOISE = 1
    IDENTIFY = 2


class SharpMode(CustomIntEnum):
    NONE = 0
    UNSHARP = 1
    UNSHARP_MINMAX = 2


class SharpLimitMode(CustomIntEnum):
    NONE = 0
    SPATIAL_PRESMOOTH = 1
    TEMPORAL_PRESMOOTH = 2
    SPATIAL_POSTSMOOTH = 3
    TEMPORAL_POSTSMOOTH = 4


class BackBlendMode(CustomIntEnum):
    NONE = 0
    PRELIMIT = 1
    POSTLIMIT = 2
    BOTH = 3


class SourceMatchMode(CustomIntEnum):
    NONE = 0
    BASIC = 1
    REFINED = 2
    TWICE_REFINED = 3
