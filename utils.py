from vstools import FunctionUtil, ConstantFormatVideoNode, vs


def scdetect(clip: vs.VideoNode, thr: float | None = None) -> ConstantFormatVideoNode:
    func = FunctionUtil(clip, scdetect, 0, (vs.GRAY, vs.YUV), (8, 16))

    props_clip = func.work_clip.misc.SCDetect(thr)

    return clip.std.CopyFrameProps(props_clip)  # type: ignore
