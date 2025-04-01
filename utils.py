from vstools import FunctionUtil, vs


def scdetect(clip: vs.VideoNode, thr: float | None = None) -> vs.VideoNode:
    func = FunctionUtil(clip, scdetect, 0, (vs.GRAY, vs.YUV))

    props_clip = func.work_clip.misc.SCDetect(thr)

    return clip.std.CopyFrameProps(props_clip, ('_SceneChangePrev', '_SceneChangeNext'))
