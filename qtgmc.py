from typing import Literal
from numpy import linalg, zeros
from math import factorial
from functools import partial
from jetpytools import normalize_seq

from enums import InputType, SearchPostProcess, NoiseProcessMode, NoiseDeintMode, SharpMode, SharpLimitMode, SourceMatchMode
from utils import scdetect

from vstools import (
    vs,
    core,
    VSFunction,
    ConvMode,
    FieldBased,
    CustomRuntimeError,
    ConstantFormatVideoNode,
    scale_delta,
    check_variable,
)
from vsrgtools import (
    MeanMode,
    BlurMatrix,
    RepairMode,
    RemoveGrainMode,
    sbr,
    repair,
    unsharpen,
    gauss_blur,
    median_blur,
    remove_grain,
)
from vsmasktools import Morpho, Coordinates
from vsdenoise import DFTTest, MVTools, MaskMode, MVDirection, MotionVectors, prefilter_to_full_range
from vsexprtools import norm_expr
from vsdeband import AddNoise
from vsaa import Eedi3, Nnedi3


class QTempGaussMC:
    """
    Quasi Temporal Gaussian Motion Compensated (QTGMC)
    """

    clip: ConstantFormatVideoNode

    def __init__(self, clip: vs.VideoNode, input_type: InputType = InputType.INTERLACE) -> None:
        assert check_variable(clip, self.__class__)

        self.clip = clip
        self.input_type = input_type

        clip_fieldbased = FieldBased.from_video(self.clip, True, self.__class__)
        self.tff = clip_fieldbased.is_tff

        if self.input_type == InputType.PROGRESSIVE and clip_fieldbased.is_inter:
            raise CustomRuntimeError(f'{self.input_type} incompatible with interlaced video!', self.__class__)

    def prefilter(
            self, clip: vs.VideoNode, tr: int = 1, sc_threshold: float | None | Literal[False] = None,
            postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
            blur_strength: tuple[float, float] = (1.9, 0.1),
            soften_limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
            range_conversion: float = 2.0
        ) -> ConstantFormatVideoNode:

        match self.input_type:
            case InputType.INTERLACE:
                search = clip.resize.Bob(tff=self.tff)
            case InputType.PROGRESSIVE:
                search = clip
            case InputType.REPAIR:
                search = BlurMatrix.BINOMIAL()(clip, mode=ConvMode.VERTICAL)

        if tr:
            scenechanges = scdetect(search, sc_threshold) if sc_threshold is not False else search
            smoothed = sbr(scenechanges, tr, ConvMode.TEMPORAL, scenechange=sc_threshold is not False)
            repaired = self.mask_shimmer(smoothed, search)
        else:
            repaired = search  # type: ignore

        if postprocess:
            gauss_sigma, blend_weight = blur_strength

            blurred = core.std.Merge(gauss_blur(repaired, gauss_sigma), repaired, blend_weight)  # type: ignore

            if postprocess == SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(_, 8, clip) for _ in soften_limit]

                blurred = norm_expr(
                    [blurred, repaired, search],
                    'z y {lim1} - y {lim1} + clip TWEAK! '
                    'x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - x 51 * y 49 * + 100 / ? ?',
                    lim1=lim1, lim2=lim2, lim3=lim3
                )  # type: ignore
        else:
            blurred = repaired

        if range_conversion:
            blurred = prefilter_to_full_range(blurred, range_conversion)  # type: ignore

        self.draft = search if self.input_type == InputType.INTERLACE else clip

        return blurred  # type: ignore

    def mask_shimmer(
            self, clip: vs.VideoNode, ref: vs.VideoNode, threshold: float | int = 1,
            erosion_distance: int = 4, over_dilation: int = 0
        ) -> ConstantFormatVideoNode:

        if not erosion_distance:
            return clip  # type: ignore

        diff = ref.std.MakeDiff(clip)

        opening = Morpho.minimum(diff, iterations=erosion_distance, coords=Coordinates.VERTICAL)
        opening = Morpho.deflate(opening, coords=Coordinates.VERTICAL)
        opening = Morpho.maximum(opening, iterations=erosion_distance, coords=Coordinates.VERTICAL)

        closing = Morpho.maximum(diff, iterations=erosion_distance, coords=Coordinates.VERTICAL)
        closing = Morpho.inflate(closing, coords=Coordinates.VERTICAL)
        closing = Morpho.minimum(closing, iterations=erosion_distance, coords=Coordinates.VERTICAL)

        if over_dilation:
            opening = Morpho.inflate(opening, over_dilation, coords=Coordinates.VERTICAL)
            closing = Morpho.deflate(closing, over_dilation, coords=Coordinates.VERTICAL)

        return norm_expr(
            [clip, diff, opening, closing],
            'y neutral - abs {thr} > y a neutral min z neutral max clip y ? neutral - x +',
            thr=scale_delta(threshold, 8, clip)
        )  # type: ignore
    
    def reinterlace(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        return clip.std.SeparateFields(self.tff).std.SelectEvery(4, (0, 3)).std.DoubleWeave(self.tff)[::2]  # type: ignore

    def denoise(
            self, restore: bool = True, tr: int = 1, stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
            denoiser: VSFunction = partial(DFTTest.denoise), process_mode: NoiseProcessMode = NoiseProcessMode.DENOISE,
            deint_mode: NoiseDeintMode = NoiseDeintMode.WEAVE,
        ) -> ConstantFormatVideoNode:

        if not process_mode:
            return self.clip

        denoised = self.mv.compensate(self.draft, tr=tr, temporal_func=denoiser) if tr else denoiser(self.draft)

        if self.input_type == InputType.INTERLACE:
            denoised = self.reinterlace(denoised)

        if restore:
            noise = self.clip.std.MakeDiff(denoised)

            if self.input_type != InputType.INTERLACE:
                noise_deint = noise
            else:
                match deint_mode:
                    case NoiseDeintMode.WEAVE:
                        noise_deint = core.std.Interleave([noise] * 2)
                    case NoiseDeintMode.BOB:
                        noise_deint = noise.resize.Bob(tff=self.tff)
                    case NoiseDeintMode.GENERATE:
                        noise_source = noise.std.SeparateFields(self.tff)

                        noise_max = Morpho.maximum(Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL)
                        noise_min = Morpho.minimum(Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL)

                        noise_new = AddNoise.GAUSS.grain(
                            noise_source, 2048, protect_chroma=False, fade_limits=False, neutral_out=True
                        )
                        noise_limit = norm_expr([noise_max, noise_min, noise_new], 'x y - z * range_size / y +')

                        noise_deint = core.std.Interleave([noise_source, noise_limit]).std.DoubleWeave(self.tff)

            if stabilize:
                weight1, weight2 = stabilize

                noise_comp = self.mv.compensate(noise_deint, direction=MVDirection.BACKWARD, tr=1, interleave=False)

                noise_deint = norm_expr(
                    [noise_deint, noise_comp],
                    'x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +',
                    weight1=weight1, weight2=weight2,
                )

            self.noise = noise_deint

        return denoised if NoiseProcessMode == NoiseProcessMode.DENOISE else self.clip

    def binomial_degrain(self, clip: vs.VideoNode, tr: int = 1, **kwargs) -> ConstantFormatVideoNode:

        def _get_weights(n: int):
            k, rhs = 1, []
            mat = zeros((n + 1, n + 1))

            for i in range(1, n + 2):
                mat[n + 1 - i, i - 1] = mat[n, i - 1] = 1 / 3
                rhs.append(k)
                k = k * (2 * n + 1 - i) // i

            mat[n, 0] = 1

            return linalg.solve(mat, rhs)

        if not tr:
            return clip  # type: ignore

        backward, forward = self.mv.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[ConstantFormatVideoNode]()

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)

            degrained.append(self.mv.degrain(clip, tr=1, vectors=vectors, **kwargs))
            vectors.clear()

        return core.std.AverageFrames([clip, *degrained], _get_weights(tr))  # type: ignore

    def source_match(
        self, clip: vs.VideoNode, ref: vs.VideoNode, tr: tuple[int, int] = (1, 1),
        similarity: float = 0.5, refine: SourceMatchMode = SourceMatchMode.BASIC, enhance: float = 0.5,
        interpolator: VSFunction = partial(Eedi3(field=3, sclip_aa=Nnedi3(field=3)).interpolate, double_y=False),
    ) -> ConstantFormatVideoNode:

        def error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:

            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2 ** tr_f / (binomial_coeff + similarity * (2 ** tr_f - binomial_coeff))

            return norm_expr([clip, ref], 'y {adj} 1 + * x {adj} * -', adj=error_adj)  # type: ignore

        basic_tr, refine_tr = normalize_seq(tr, 2)

        if not refine:
            return clip  # type: ignore

        if self.input_type != InputType.PROGRESSIVE:
            clip = self.reinterlace(clip)

        adjusted1 = error_adjustment(clip, ref, basic_tr)
        bobbed1 = interpolator(adjusted1)
        match1 = self.binomial_degrain(bobbed1, basic_tr)

        if refine > SourceMatchMode.BASIC:
            if enhance:
                match1 = unsharpen(match1, enhance, BlurMatrix.BINOMIAL())

            if self.input_type != InputType.PROGRESSIVE:
                match1 = self.reinterlace(match1)

            diff = ref.std.MakeDiff(match1)
            bobbed2 = interpolator(diff)
            match2 = self.binomial_degrain(bobbed2, refine_tr)

            if refine == SourceMatchMode.TWICE_REFINED:
                adjusted2 = error_adjustment(match2, bobbed2, refine_tr)
                match2 = self.binomial_degrain(adjusted2, refine_tr)

            out = match1.std.MergeDiff(match2)
        else:
            out = match1

        return out  # type: ignore

    def lossless(self, clip: vs.VideoNode, ref: vs.VideoNode) -> ConstantFormatVideoNode:

        def _reweave(clipa: vs.VideoNode, clipb: vs.VideoNode) -> ConstantFormatVideoNode:
            return (
                core.std.Interleave([clipa, clipb])  # type: ignore
                .std.SelectEvery(4, (0, 1, 3, 2))
                .std.DoubleWeave(self.tff)[::2]
            )

        if self.input_type == InputType.PROGRESSIVE:
            raise CustomRuntimeError(f'{self.lossless} incompatible with {self.input_type}!', self.lossless)

        fields_ref = ref.std.SeparateFields(self.tff)

        if self.input_type == InputType.REPAIR:
            fields_ref = fields_ref.std.SelectEvery(4, (0, 3))

        fields_new = clip.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = _reweave(fields_ref, fields_new)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = core.std.Expr(
            [median_blur(fields_diff, mode=ConvMode.VERTICAL), fields_diff],
            'x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?'
        )
        processed_diff = repair(
            processed_diff, remove_grain(processed_diff, RemoveGrainMode.MINMAX_AROUND2), RepairMode.MINMAX_SQUARE1
        )

        return _reweave(fields_ref, core.std.MakeDiff(fields_new, processed_diff))  # type: ignore

    def sharpen(
        self, clip: vs.VideoNode, sharp: SharpMode = SharpMode.NONE,
        strength: float = 0.0, limit: int | float = 1, thin: float = 0.0
    ) -> ConstantFormatVideoNode:

        blur_kernel = BlurMatrix.BINOMIAL()

        match sharp:
            case SharpMode.NONE:
                resharp = clip
            case SharpMode.UNSHARP:
                resharp = unsharpen(clip, strength, blur_kernel)
            case SharpMode.UNSHARP_MINMAX:
                source_min = Morpho.minimum(clip, coords=Coordinates.HORIZONTAL)
                source_max = Morpho.maximum(clip, coords=Coordinates.HORIZONTAL)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    'y z + 2 / AVG! x AVG@ {thr} - AVG@ {thr} + clip',
                    thr=scale_delta(limit, 8, clip),
                )
                resharp = unsharpen(clip, strength, blur_kernel(clamp))

        if thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)], 'y x - {thin} * neutral +', thin=thin
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff)

            resharp = norm_expr(
                [resharp, blurred_diff, blur_kernel(blurred_diff)], 'y neutral - abs z neutral - abs < y neutral ? x +'
            )

        return resharp  # type: ignore

    def sharp_limit(
        self, clip: vs.VideoNode, ref: vs.VideoNode,
        mode: SharpLimitMode = SharpLimitMode.TEMPORAL_PRESMOOTH,
        radius: int = 1, limit: int | float = 0
    ) -> ConstantFormatVideoNode:

        if mode:
            if mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.SPATIAL_POSTSMOOTH):
                if radius == 1:
                    clip = repair(clip, ref, RepairMode.MINMAX_SQUARE1)
                elif radius > 1:
                    clip = repair(clip, repair(clip, ref, RepairMode.MINMAX_SQUARE_REF2), RepairMode.MINMAX_SQUARE1)

            if mode in (SharpLimitMode.TEMPORAL_PRESMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
                backward, forward = self.mv.compensate(ref, tr=radius, interleave=False)

                comp_min = MeanMode.MINIMUM([ref, *backward, *forward])
                comp_max = MeanMode.MAXIMUM([ref, *backward, *forward])

                clip = norm_expr(
                    [clip, comp_min, comp_max], 'x y {limit} - z {limit} + clip', limit=scale_delta(limit, 8, clip)
                )

        return clip  # type: ignore

    def back_blend(self, clip: vs.VideoNode, ref: vs.VideoNode, sigma: float = 1.4) -> ConstantFormatVideoNode:

        if sigma:
            clip = clip.std.MakeDiff(gauss_blur(clip.std.MakeDiff(ref), sigma))

        return clip  # type: ignore

    def restore_noise(self, clip: vs.VideoNode, restore: float = 0.0) -> ConstantFormatVideoNode:

        if restore and self.noise:
            clip = norm_expr([clip, self.noise], 'y neutral - {restore} * x +', restore=restore)

        return clip  # type: ignore

    def motion_blur(
            self, clip: vs.VideoNode, shutter_angle: tuple[int, int],
            shutter_blur: tuple[int, float | None | Literal[False]] | Literal[False] = False,
            fps_divisor: int = 1, search: vs.VideoNode | None = None
        ) -> ConstantFormatVideoNode:

        if shutter_blur is not False:
            angle_src, angle_out = shutter_angle
            blur_amount, blur_limit = shutter_blur

            blur_level = (angle_out * fps_divisor - angle_src) * 100 / 360

            if blur_amount:
                processed = self.mv.flow_blur(clip, blur=blur_level)

                if blur_limit is not False:
                    mask = self.mv.mask(search, direction=MVDirection.BACKWARD, kind=MaskMode.MOTION, ml=blur_limit)

                    processed = clip.std.MaskedMerge(processed, mask)
        else:
            processed = clip

        if fps_divisor > 1:
            processed = processed[::fps_divisor]

        return processed

    def process(self):
        search = self.prefilter(self.clip)


        self.mv = MVTools(self.draft, search)
