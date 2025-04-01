from typing import Literal
from numpy import linalg, zeros
from math import factorial
from functools import partial

from enums import InputType, SearchPostProcess, NoiseProcessMode, NoiseDeintMode, SharpMode, SharpLimitMode, SourceMatchMode, LosslessMode, BackBlendMode
from utils import scdetect

from vstools import (
    vs,
    core,
    VSFunction,
    FieldBasedT,
    ConvMode,
    FieldBased,
    CustomRuntimeError,
    ConstantFormatVideoNode,
    scale_delta,
    check_variable,
    fallback
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
from vsdenoise import DFTTest, MVTools, MaskMode, MVDirection, MotionVectors, prefilter_to_full_range, MVToolsPresets, MVToolsPreset
from vsexprtools import norm_expr
from vsdeband import AddNoise
from vsaa import Nnedi3
from vsdeinterlace import reinterlace


class QTempGaussMC:
    """
    Quasi Temporal Gaussian Motion Compensated (QTGMC)
    """

    clip: ConstantFormatVideoNode

    def __init__(
            self,
            clip: vs.VideoNode,
            input_type: InputType = InputType.INTERLACE,
            tff: FieldBasedT | bool | None = None,
            preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
            lossless_mode: LosslessMode = LosslessMode.PRESHARPEN,
            force_tr: int = 0,

            # prefilter stage args
            prefilter_tr: int = 1,
            prefilter_sc_threshold: float | None | Literal[False] = None,
            prefilter_postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
            prefilter_blur_strength: tuple[float, float] = (1.9, 0.1),
            prefilter_soften_limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
            prefilter_range_conversion: float = 2.0,
            prefilter_threshold: float | int = 1,
            prefilter_erosion_distance: int = 4,
            prefilter_over_dilation: int = 0,

            # denoise stage args
            denoise_tr: int = 1,
            denoise_restore: bool = True,
            denoise_stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
            denoise_func: VSFunction = partial(DFTTest.denoise),
            denoise_deint: NoiseDeintMode = NoiseDeintMode.WEAVE,
            denoise_mode: NoiseProcessMode = NoiseProcessMode.BYPASS,

            # basic stage args
            basic_interpolator: VSFunction = partial(Nnedi3(field=3).interpolate, double_y=False),
            basic_tr: int = 1,
            basic_threshold: float | int = 1,
            basic_erosion_distance: int = 4,
            basic_over_dilation: int = 0,
            basic_noise_restore: float = 0,

            # source match stage args
            match_mode: SourceMatchMode = SourceMatchMode.BASIC,
            match_interpolator: VSFunction | None = None,
            match_tr: int = 1,
            match_similarity: float = 0.5,
            match_enhance: float = 0.5,

            # sharpen stage
            sharp_mode: SharpMode = SharpMode.NONE,
            sharp_strength: float = 0.0,
            sharp_clamp: int | float = 1,
            sharp_thin: float = 0.0,

            # back blending
            backblend_mode: BackBlendMode = BackBlendMode.NONE,
            backblend_sigma: float = 1.4,

            # sharp limiting stage
            sharplimit_mode: SharpLimitMode = SharpLimitMode.NONE,
            sharplimit_radius: int = 1,
            sharplimit_limit: int | float = 0,

            # final stage
            final_tr: int = 1,
            final_noise_restore: float = 0.0,

            # motion blur stage
            motion_blur_shutter_angle: tuple[int, int] | Literal[False] = False,
            motion_blur_limit: float | None | Literal[False] = None,
            motion_blur_fps_divisor: int = 1,
        ) -> None:

        assert check_variable(clip, self.__class__)

        self.clip = clip
        self.input_type = input_type

        clip_fieldbased = FieldBased.from_param_or_video(tff, self.clip, True, self.__class__)
        self.tff = clip_fieldbased.is_tff

        if self.input_type == InputType.PROGRESSIVE and clip_fieldbased.is_inter:
            raise CustomRuntimeError(f'{self.input_type} incompatible with interlaced video!', self.__class__)
        
        preset.pop('search_clip', None)

        self.preset = preset
        self.lossless_mode = lossless_mode
        self.force_tr = force_tr

        self.prefilter_tr = prefilter_tr
        self.prefilter_sc_threshold = prefilter_sc_threshold
        self.prefilter_postprocess = prefilter_postprocess
        self.prefilter_blur_strength = prefilter_blur_strength
        self.prefilter_soften_limit = prefilter_soften_limit
        self.prefilter_range_conversion = prefilter_range_conversion
        self.prefilter_threshold = prefilter_threshold
        self.prefilter_erosion_distance = prefilter_erosion_distance
        self.prefilter_over_dilation = prefilter_over_dilation

        self.denoise_tr = denoise_tr
        self.denoise_restore = denoise_restore
        self.denoise_stabilize = denoise_stabilize
        self.denoise_func = denoise_func
        self.denoise_deint = denoise_deint
        self.denoise_mode = denoise_mode

        self.basic_interpolator = basic_interpolator
        self.basic_tr = basic_tr
        self.basic_threshold = basic_threshold
        self.basic_erosion_distance = basic_erosion_distance
        self.basic_over_dilation = basic_over_dilation
        self.basic_noise_restore = basic_noise_restore

        self.match_mode = match_mode
        self.match_interpolator = fallback(match_interpolator, self.basic_interpolator)
        self.match_tr = match_tr
        self.match_similarity = match_similarity
        self.match_enhance = match_enhance

        self.sharp_mode = sharp_mode
        self.sharp_strength = sharp_strength
        self.sharp_clamp = sharp_clamp
        self.sharp_thin = sharp_thin

        self.backblend_mode = backblend_mode
        self.backblend_sigma = backblend_sigma

        self.sharplimit_mode = sharplimit_mode
        self.sharplimit_radius = sharplimit_radius
        self.sharplimit_limit = sharplimit_limit

        self.final_tr = final_tr
        self.final_noise_restore = final_noise_restore

        self.motion_blur_shutter_angle = motion_blur_shutter_angle
        self.motion_blur_limit = motion_blur_limit
        self.motion_blur_fps_divisor = motion_blur_fps_divisor

    def mask_shimmer(
            self, clip: vs.VideoNode, ref: vs.VideoNode, threshold: float | int,
            erosion_distance: int, over_dilation: int
        ) -> ConstantFormatVideoNode:

        if not erosion_distance:
            return clip

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
        )

    def binomial_degrain(self, clip: vs.VideoNode, tr: int, **kwargs) -> ConstantFormatVideoNode:

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
            return clip

        backward, forward = self.mv.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[ConstantFormatVideoNode]()

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)
            vectors.tr = 1

            degrained.append(self.mv.degrain(clip, tr=1, vectors=vectors, **kwargs))
            vectors.clear()

        return core.std.AverageFrames([clip, *degrained], _get_weights(tr))

    def prefilter(self) -> ConstantFormatVideoNode:

        match self.input_type:
            case InputType.INTERLACE:
                search = self.clip.resize.Bob(tff=self.tff)
            case InputType.PROGRESSIVE:
                search = self.clip
            case InputType.REPAIR:
                search = BlurMatrix.BINOMIAL()(self.clip, mode=ConvMode.VERTICAL)

        if self.prefilter_tr:
            scenechange = self.prefilter_sc_threshold is not False

            scenechanges = scdetect(search, self.prefilter_sc_threshold) if scenechange else search
            smoothed = sbr(scenechanges, self.prefilter_tr, ConvMode.TEMPORAL, scenechange=scenechange)
            repaired = self.mask_shimmer(
                smoothed, search,
                self.prefilter_threshold,
                self.prefilter_erosion_distance,
                self.prefilter_over_dilation,
            )
        else:
            repaired = search

        if self.prefilter_postprocess:
            gauss_sigma, blend_weight = self.prefilter_blur_strength

            blurred = core.std.Merge(gauss_blur(repaired, gauss_sigma), repaired, blend_weight)

            if self.prefilter_postprocess == SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(_, 8, self.clip) for _ in self.prefilter_soften_limit]

                blurred = norm_expr(
                    [blurred, repaired, search],
                    'z y {lim1} - y {lim1} + clip TWEAK! '
                    'x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - x 51 * y 49 * + 100 / ? ?',
                    lim1=lim1, lim2=lim2, lim3=lim3
                )
        else:
            blurred = repaired

        if self.prefilter_range_conversion:
            blurred = prefilter_to_full_range(blurred, self.prefilter_range_conversion)

        self.draft = search if self.input_type == InputType.INTERLACE else self.clip

        return blurred

    def denoise(self) -> ConstantFormatVideoNode:

        if not self.denoise_mode:
            return self.clip

        if self.denoise_tr:
            denoised = self.mv.compensate(self.draft, tr=self.denoise_tr, temporal_func=self.denoise_func)
        else:
            denoised = self.denoise_func(self.draft)

        if self.input_type == InputType.INTERLACE:
            denoised = reinterlace(denoised, self.tff)

        if self.denoise_restore:
            noise = self.clip.std.MakeDiff(denoised)

            if self.input_type != InputType.INTERLACE:
                noise_deint = noise
            else:
                match self.denoise_deint:
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

            if self.denoise_stabilize:
                weight1, weight2 = self.denoise_stabilize

                noise_comp, _ = self.mv.compensate(noise_deint, direction=MVDirection.BACKWARD, tr=1, interleave=False)

                noise_deint = norm_expr(
                    [noise_deint, *noise_comp],
                    'x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +',
                    weight1=weight1, weight2=weight2,
                )

            self.noise = noise_deint

        return denoised if self.denoise_mode == NoiseProcessMode.DENOISE else self.clip

    def source_match(self, clip: vs.VideoNode, ref: vs.VideoNode) -> ConstantFormatVideoNode:

        def error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:

            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2 ** tr_f / (binomial_coeff + self.match_similarity * (2 ** tr_f - binomial_coeff))

            return norm_expr([clip, ref], 'y {adj} 1 + * x {adj} * -', adj=error_adj)

        if not self.match_mode:
            return clip

        if self.input_type != InputType.PROGRESSIVE:
            clip = reinterlace(clip, self.tff)

        adjusted1 = error_adjustment(clip, ref, self.basic_tr)
        bobbed1 = self.basic_interpolator(adjusted1)
        match1 = self.binomial_degrain(bobbed1, self.basic_tr)

        if self.match_mode > SourceMatchMode.BASIC:
            if self.match_enhance:
                match1 = unsharpen(match1, self.match_enhance, BlurMatrix.BINOMIAL())

            if self.input_type != InputType.PROGRESSIVE:
                match1 = reinterlace(match1, self.tff)

            diff = ref.std.MakeDiff(match1)
            bobbed2 = self.match_interpolator(diff)
            match2 = self.binomial_degrain(bobbed2, self.match_tr)

            if self.match_mode == SourceMatchMode.TWICE_REFINED:
                adjusted2 = error_adjustment(match2, bobbed2, self.match_tr)
                match2 = self.binomial_degrain(adjusted2, self.match_tr)

            out = match1.std.MergeDiff(match2)
        else:
            out = match1

        return out

    def lossless(self, clip: vs.VideoNode, ref: vs.VideoNode) -> ConstantFormatVideoNode:

        def _reweave(clipa: vs.VideoNode, clipb: vs.VideoNode) -> ConstantFormatVideoNode:
            return (
                core.std.Interleave([clipa, clipb])
                .std.SelectEvery(4, (0, 1, 3, 2))
                .std.DoubleWeave(self.tff)[::2]
            )

        if self.input_type == InputType.PROGRESSIVE:
            return clip

        fields_ref = ref.std.SeparateFields(self.tff)

        if self.input_type == InputType.REPAIR:
            fields_ref = fields_ref.std.SelectEvery(4, (0, 3))

        fields_new = clip.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = _reweave(fields_ref, fields_new)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = norm_expr(
            [median_blur(fields_diff, mode=ConvMode.VERTICAL), fields_diff],
            'x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?'
        )
        processed_diff = repair(
            processed_diff, remove_grain(processed_diff, RemoveGrainMode.MINMAX_AROUND2), RepairMode.MINMAX_SQUARE1
        )

        return _reweave(fields_ref, core.std.MakeDiff(fields_new, processed_diff))

    def sharpen(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:

        blur_kernel = BlurMatrix.BINOMIAL()

        match self.sharp_mode:
            case SharpMode.NONE:
                resharp = clip
            case SharpMode.UNSHARP:
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel)
            case SharpMode.UNSHARP_MINMAX:
                source_min = Morpho.minimum(clip, coords=Coordinates.HORIZONTAL)
                source_max = Morpho.maximum(clip, coords=Coordinates.HORIZONTAL)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    'y z + 2 / AVG! x AVG@ {thr} - AVG@ {thr} + clip',
                    thr=scale_delta(self.sharp_clamp, 8, clip),
                )
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel(clamp))

        if self.sharp_thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)], 'y x - {thin} * neutral +', thin=self.sharp_thin
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff)

            resharp = norm_expr(
                [resharp, blurred_diff, blur_kernel(blurred_diff)],
                'y neutral - Y! z neutral - Z! Y@ abs Z@ abs < Y@ 0 ? x +',
            )

        return resharp

    def sharp_limit(self, clip: vs.VideoNode, ref: vs.VideoNode,) -> ConstantFormatVideoNode:

        if self.sharplimit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.SPATIAL_POSTSMOOTH):
            if self.sharplimit_radius == 1:
                clip = repair(clip, ref, RepairMode.MINMAX_SQUARE1)
            elif self.sharplimit_radius > 1:
                clip = repair(clip, repair(clip, ref, RepairMode.MINMAX_SQUARE_REF2), RepairMode.MINMAX_SQUARE1)

        if self.sharplimit_mode in (SharpLimitMode.TEMPORAL_PRESMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
            backward, forward = self.mv.compensate(ref, tr=self.sharplimit_radius, interleave=False)

            comp_min = MeanMode.MINIMUM([ref, *backward, *forward])
            comp_max = MeanMode.MAXIMUM([ref, *backward, *forward])

            clip = norm_expr(
                [clip, comp_min, comp_max], 'x y {thr} - z {thr} + clip', thr=scale_delta(self.sharplimit_limit, 8, clip)
            )

        return clip

    def back_blend(self, clip: vs.VideoNode, ref: vs.VideoNode) -> ConstantFormatVideoNode:

        if self.backblend_sigma:
            clip = clip.std.MakeDiff(gauss_blur(clip.std.MakeDiff(ref), self.backblend_sigma))

        return clip

    def restore_noise(self, clip: vs.VideoNode, restore: float = 0.0) -> ConstantFormatVideoNode:

        if restore and self.noise:
            clip = norm_expr([clip, self.noise], 'y neutral - {restore} * x +', restore=restore)

        return clip

    def motion_blur(self, clip: vs.VideoNode, search: vs.VideoNode) -> ConstantFormatVideoNode:

        if self.motion_blur_shutter_angle is not False:
            angle_src, angle_out = self.motion_blur_shutter_angle

            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_src) * 100 / 360

            processed = self.mv.flow_blur(clip, blur=blur_level)

            if self.motion_blur_limit is not False:
                mask = self.mv.mask(search, direction=MVDirection.BACKWARD, kind=MaskMode.MOTION, ml=self.motion_blur_limit)

                processed = clip.std.MaskedMerge(processed, mask)
        else:
            processed = clip

        if self.motion_blur_fps_divisor > 1:
            processed = processed[::self.motion_blur_fps_divisor]

        return processed

    def basic(self, bobbed: vs.VideoNode, denoised: vs.VideoNode) -> ConstantFormatVideoNode:

        smoothed = self.binomial_degrain(bobbed, tr=self.basic_tr)

        masked = self.mask_shimmer(
            smoothed, bobbed, self.basic_threshold, self.basic_erosion_distance, self.basic_over_dilation
        )

        matched = self.source_match(masked, bobbed)

        if self.lossless_mode == LosslessMode.PRESHARPEN:
            matched = self.lossless(matched, denoised)

        sharp = self.sharpen(matched)

        if self.backblend_mode in (BackBlendMode.PRELIMIT, BackBlendMode.BOTH):
            sharp = self.back_blend(sharp, matched)

        if self.sharp_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.TEMPORAL_PRESMOOTH):
            sharp = self.sharp_limit(sharp, bobbed)

        if self.backblend_mode in (BackBlendMode.POSTLIMIT, BackBlendMode.BOTH):
            sharp = self.back_blend(sharp, matched)

        return self.restore_noise(sharp, self.basic_noise_restore)

    def final(self, basic: vs.VideoNode, bobbed: vs.VideoNode, denoised: vs.VideoNode) -> ConstantFormatVideoNode:

        smoothed = self.mv.degrain(basic, tr=self.final_tr)

        masked = self.mask_shimmer(
            smoothed, bobbed, self.basic_threshold, self.basic_erosion_distance, self.basic_over_dilation
        )

        if self.sharp_mode in (SharpLimitMode.SPATIAL_POSTSMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
            masked = self.sharp_limit(masked, bobbed)

        if self.lossless_mode == LosslessMode.POSTSMOOTH:
            masked = self.lossless(masked, denoised)

        return self.restore_noise(masked, self.final_noise_restore)

    def process(self) -> ConstantFormatVideoNode:

        tr = max(self.force_tr, self.denoise_tr, self.basic_tr, self.match_tr, self.final_tr)

        search = self.prefilter()

        self.mv = MVTools(self.draft, search, **self.preset)
        self.mv.analyze(tr=tr)

        denoised = self.denoise()

        if self.input_type == InputType.REPAIR:
            denoised = reinterlace(denoised, self.tff)

        bobbed = self.basic_interpolator(denoised)

        if self.input_type == InputType.REPAIR:
            mask = self.mv.mask(search, direction=MVDirection.BACKWARD)
            bobbed = denoised.std.MaskedMerge(bobbed, mask)

        basic_out = self.basic(bobbed, denoised)
        final_out = self.final(basic_out, bobbed, denoised)
        blurred = self.motion_blur(final_out, search)

        return blurred
