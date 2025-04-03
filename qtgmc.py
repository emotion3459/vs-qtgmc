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
    fallback,
    KwargsT
)
from vsrgtools import (
    MeanMode,
    BlurMatrix,
    RepairMode,
    RemoveGrainMode,
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

            # global stuff
            preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
            lossless_mode: LosslessMode = LosslessMode.NONE,
            force_tr: int = 0,
            show_noise: bool = False,

            # prefilter stage
            prefilter_tr: int = 2,
            prefilter_sc_threshold: float | None | Literal[False] = None,
            prefilter_postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
            prefilter_blur_strength: tuple[float, float] = (1.9, 0.1),
            prefilter_soften_limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
            prefilter_range_conversion: float = 2.0,
            prefilter_threshold: float | int = 1,
            prefilter_erosion_distance: int = 4,
            prefilter_over_dilation: int = 0,

            # denoise stage
            denoise_tr: int = 2,
            denoise_stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
            denoise_func: VSFunction = partial(DFTTest.denoise),  # fixme
            denoise_deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
            denoise_mode: NoiseProcessMode = NoiseProcessMode.IDENTIFY,
            denoise_func_comp_args: KwargsT = KwargsT(),
            denoise_stabilize_comp_args: KwargsT = KwargsT(),

            # basic stage
            basic_interpolator: VSFunction = partial(core.znedi3.nnedi3, field=3, qual=2, nsize=0, nns=4, pscrn=1),  # fixme
            basic_tr: int = 2,
            basic_threshold: float | int = 1,
            basic_erosion_distance: int = 0,
            basic_over_dilation: int = 0,
            basic_noise_restore: float = 0,
            basic_degrain_args: KwargsT = KwargsT(),

            # source match stage
            match_mode: SourceMatchMode = SourceMatchMode.NONE,
            match_interpolator: VSFunction | None = None,
            match_tr: int = 1,
            match_similarity: float = 0.5,
            match_enhance: float = 0.5,

            # sharpen stage
            sharp_mode: SharpMode = SharpMode.UNSHARP_MINMAX,
            sharp_strength: float = 1.0,
            sharp_clamp: int | float = 1,
            sharp_thin: float = 0.0,

            # back blending
            backblend_mode: BackBlendMode = BackBlendMode.BOTH,
            backblend_sigma: float = 1.4,

            # sharp limiting stage
            sharplimit_mode: SharpLimitMode = SharpLimitMode.TEMPORAL_PRESMOOTH,
            sharplimit_radius: int = 3,
            sharplimit_limit: int | float = 0,
            sharplimit_comp_args: KwargsT = KwargsT(),

            # final stage
            final_tr: int = 3,
            final_noise_restore: float = 0.0,
            final_threshold: float | int = 1,
            final_erosion_distance: int = 4,
            final_over_dilation: int = 0,
            final_degrain_args: KwargsT = KwargsT(),

            # motion blur stage
            motion_blur_shutter_angle: tuple[int | float, int | float] = (180, 180),
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
        self.show_noise = show_noise

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
        self.denoise_stabilize = denoise_stabilize
        self.denoise_func = denoise_func
        self.denoise_deint = denoise_deint
        self.denoise_mode = denoise_mode
        self.denoise_func_comp_args = denoise_func_comp_args
        self.denoise_stabilize_comp_args = denoise_stabilize_comp_args

        self.basic_interpolator = basic_interpolator
        self.basic_tr = basic_tr
        self.basic_threshold = basic_threshold
        self.basic_erosion_distance = basic_erosion_distance
        self.basic_over_dilation = basic_over_dilation
        self.basic_noise_restore = basic_noise_restore
        self.basic_degrain_args = basic_degrain_args

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
        self.sharplimit_comp_args = sharplimit_comp_args

        self.final_tr = final_tr
        self.final_noise_restore = final_noise_restore
        self.final_erosion_distance = final_erosion_distance
        self.final_over_dilation = final_over_dilation
        self.final_threshold = final_threshold
        self.final_degrain_args = final_degrain_args

        self.motion_blur_shutter_angle = motion_blur_shutter_angle
        self.motion_blur_limit = motion_blur_limit
        self.motion_blur_fps_divisor = motion_blur_fps_divisor

    def _mask_shimmer(
            self, flt: vs.VideoNode, src: vs.VideoNode, threshold: float | int,
            erosion_distance: int, over_dilation: int
        ) -> ConstantFormatVideoNode:

        if not erosion_distance:
            return flt

        iter1 = 1 + (erosion_distance + 1) // 3
        iter2 = 1 + (erosion_distance + 2) // 3

        diff = src.std.MakeDiff(flt)

        opening = Morpho.minimum(diff, iterations=iter1, coords=Coordinates.VERTICAL)

        if erosion_distance % 3:
            opening = Morpho.deflate(opening)
            if erosion_distance % 3 == 2:
                opening = median_blur(opening)

        opening = Morpho.maximum(opening, iterations=iter2, coords=Coordinates.VERTICAL)

        closing = Morpho.maximum(diff, iterations=iter1, coords=Coordinates.VERTICAL)

        if erosion_distance % 3:
            closing = Morpho.inflate(closing)
            if erosion_distance % 3 == 2:
                closing = median_blur(closing)

        closing = Morpho.minimum(closing, iterations=iter2, coords=Coordinates.VERTICAL)

        if over_dilation:
            opening = Morpho.maximum(iterations=over_dilation // 3)
            opening = Morpho.inflate(iterations=over_dilation % 3)

            closing = Morpho.minimum(iterations=over_dilation // 3)
            closing = Morpho.deflate(iterations=over_dilation % 3)

        return norm_expr(
            [flt, diff, opening, closing],
            'y neutral - abs {thr} > y a neutral min z neutral max clip y ? neutral - x +',
            thr=scale_delta(threshold, 8, flt)
        )

    def _binomial_degrain(self, clip: vs.VideoNode, tr: int, **kwargs) -> ConstantFormatVideoNode:

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

    def _prefilter(self, clip: vs.VideoNode) -> tuple[ConstantFormatVideoNode, ConstantFormatVideoNode]:

        match self.input_type:
            case InputType.INTERLACE:
                search = clip.resize.Bob(tff=self.tff)
            case InputType.PROGRESSIVE:
                search = clip
            case InputType.REPAIR:
                search = BlurMatrix.BINOMIAL()(clip, mode=ConvMode.VERTICAL)

        if self.prefilter_tr:
            scenechange = self.prefilter_sc_threshold is not False

            scenechanges = scdetect(search, self.prefilter_sc_threshold) if scenechange else search
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL, scenechange=scenechange)(
                scenechanges
            )
            repaired = self._mask_shimmer(
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
                lim1, lim2, lim3 = [scale_delta(_, 8, clip) for _ in self.prefilter_soften_limit]

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

        draft = search if self.input_type == InputType.INTERLACE else clip

        return (blurred, draft)

    def _denoise(self, draft: vs.VideoNode, src: vs.VideoNode) -> ConstantFormatVideoNode:

        if not self.denoise_mode:
            return src

        if self.denoise_tr:
            denoised = self.mv.compensate(
                draft, tr=self.denoise_tr, temporal_func=self.denoise_func, **self.denoise_func_comp_args
            )
        else:
            denoised = self.denoise_func(draft)

        if self.input_type == InputType.INTERLACE:
            denoised = reinterlace(denoised, self.tff)

        noise = src.std.MakeDiff(denoised)

        if self.basic_noise_restore or self.final_noise_restore:
            if self.input_type == InputType.INTERLACE:
                match self.denoise_deint:
                    case NoiseDeintMode.WEAVE:
                        noise = core.std.Interleave([noise] * 2)
                    case NoiseDeintMode.BOB:
                        noise = noise.resize.Bob(tff=self.tff)
                    case NoiseDeintMode.GENERATE:
                        noise_source = noise.std.SeparateFields(self.tff)

                        noise_max = Morpho.maximum(Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL)
                        noise_min = Morpho.minimum(Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL)

                        noise_new = AddNoise.GAUSS.grain(
                            noise_source, 2048, protect_chroma=False, fade_limits=False, neutral_out=True
                        )
                        noise_limit = norm_expr([noise_max, noise_min, noise_new], 'x y - z * range_size / y +')

                        noise = core.std.Interleave([noise_source, noise_limit]).std.DoubleWeave(self.tff)

            if self.denoise_stabilize:
                weight1, weight2 = self.denoise_stabilize

                noise_comp, _ = self.mv.compensate(
                    noise, direction=MVDirection.BACKWARD,
                    tr=1, interleave=False,
                    **self.denoise_stabilize_comp_args,
                )

                noise = norm_expr(
                    [noise, *noise_comp],
                    'x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +',
                    weight1=weight1, weight2=weight2,
                )

        self.noise = noise

        return denoised if self.denoise_mode == NoiseProcessMode.DENOISE else src

    def _source_match(self, clip: vs.VideoNode, ref: vs.VideoNode) -> ConstantFormatVideoNode:

        def _error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:

            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2 ** tr_f / (binomial_coeff + self.match_similarity * (2 ** tr_f - binomial_coeff))

            return norm_expr([clip, ref], 'y {adj} 1 + * x {adj} * -', adj=error_adj)

        if not self.match_mode:
            return clip

        if self.input_type != InputType.PROGRESSIVE:
            clip = reinterlace(clip, self.tff)

        adjusted1 = _error_adjustment(clip, ref, self.basic_tr)
        bobbed1 = self.basic_interpolator(adjusted1)
        match1 = self._binomial_degrain(bobbed1, self.basic_tr, **self.basic_degrain_args)

        if self.match_mode > SourceMatchMode.BASIC:
            if self.match_enhance:
                match1 = unsharpen(match1, self.match_enhance, BlurMatrix.BINOMIAL())

            if self.input_type != InputType.PROGRESSIVE:
                clip = reinterlace(match1, self.tff)

            diff = ref.std.MakeDiff(clip)
            bobbed2 = self.match_interpolator(diff)
            match2 = self._binomial_degrain(bobbed2, self.match_tr)

            if self.match_mode == SourceMatchMode.TWICE_REFINED:
                adjusted2 = _error_adjustment(match2, bobbed2, self.match_tr)
                match2 = self._binomial_degrain(adjusted2, self.match_tr)

            out = match1.std.MergeDiff(match2)
        else:
            out = match1

        return out

    def _lossless(self, flt: vs.VideoNode, src: vs.VideoNode) -> ConstantFormatVideoNode:

        def _reweave(clipa: vs.VideoNode, clipb: vs.VideoNode) -> ConstantFormatVideoNode:
            return (
                core.std.Interleave([clipa, clipb])
                .std.SelectEvery(4, (0, 1, 3, 2))
                .std.DoubleWeave(self.tff)[::2]
            )

        if self.input_type == InputType.PROGRESSIVE:
            return flt

        fields_src = src.std.SeparateFields(self.tff)

        if self.input_type == InputType.REPAIR:
            fields_src = fields_src.std.SelectEvery(4, (0, 3))

        fields_flt = flt.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = _reweave(fields_src, fields_flt)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = norm_expr(
            [fields_diff, median_blur(fields_diff, mode=ConvMode.VERTICAL)],
            'x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?'
        )
        processed_diff = repair(
            processed_diff, remove_grain(processed_diff, RemoveGrainMode.MINMAX_AROUND2), RepairMode.MINMAX_SQUARE1
        )

        return _reweave(fields_src, core.std.MakeDiff(fields_flt, processed_diff))

    def _sharpen(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:

        blur_kernel = BlurMatrix.BINOMIAL()

        match self.sharp_mode:
            case SharpMode.NONE:
                resharp = clip
            case SharpMode.UNSHARP:
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel)
            case SharpMode.UNSHARP_MINMAX:
                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL)

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

    def _sharp_limit(self, resharp: vs.VideoNode, bobbed: vs.VideoNode,) -> ConstantFormatVideoNode:

        if self.sharp_mode:
            if self.sharplimit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.SPATIAL_POSTSMOOTH):
                if self.sharplimit_radius == 1:
                    resharp = repair(resharp, bobbed, RepairMode.MINMAX_SQUARE1)
                elif self.sharplimit_radius > 1:
                    resharp = repair(resharp, repair(resharp, bobbed, RepairMode.MINMAX_SQUARE_REF2), RepairMode.MINMAX_SQUARE1)

            if self.sharplimit_mode in (SharpLimitMode.TEMPORAL_PRESMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
                backward_comp, forward_comp = self.mv.compensate(
                    bobbed, tr=self.sharplimit_radius, interleave=False, **self.sharplimit_comp_args
                )

                comp_min = MeanMode.MINIMUM([bobbed, *backward_comp, *forward_comp])
                comp_max = MeanMode.MAXIMUM([bobbed, *backward_comp, *forward_comp])

                resharp = norm_expr(
                    [resharp, comp_min, comp_max],
                    'x y {thr} - z {thr} + clip',
                    thr=scale_delta(self.sharplimit_limit, 8, resharp),
                )

        return resharp

    def _back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> ConstantFormatVideoNode:

        if self.backblend_sigma:
            flt = flt.std.MakeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def _restore_noise(self, clip: vs.VideoNode, restore: float = 0.0) -> ConstantFormatVideoNode:

        if restore and self.noise:
            clip = norm_expr([clip, self.noise], 'y neutral - {restore} * x +', restore=restore)

        return clip

    def _motion_blur(self, clip: vs.VideoNode, search_clip: vs.VideoNode) -> ConstantFormatVideoNode:

        angle_in, angle_out = self.motion_blur_shutter_angle

        if not angle_out * self.motion_blur_fps_divisor == angle_in:
            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

            processed = self.mv.flow_blur(clip, blur=blur_level)

            if self.motion_blur_limit is not False:
                mask = self.mv.mask(search_clip, direction=MVDirection.BACKWARD, kind=MaskMode.MOTION, ml=self.motion_blur_limit)

                processed = clip.std.MaskedMerge(processed, mask)
        else:
            processed = clip

        if self.motion_blur_fps_divisor > 1:
            processed = processed[::self.motion_blur_fps_divisor]

        return processed

    def _basic(self, bobbed: vs.VideoNode, denoised: vs.VideoNode) -> ConstantFormatVideoNode:

        smoothed = self._binomial_degrain(bobbed, tr=self.basic_tr, **self.basic_degrain_args)

        masked = self._mask_shimmer(
            smoothed, bobbed, self.basic_threshold, self.basic_erosion_distance, self.basic_over_dilation
        )

        matched = self._source_match(masked, denoised)

        if self.lossless_mode == LosslessMode.PRESHARPEN:
            matched = self._lossless(matched, denoised)

        resharp = self._sharpen(matched)

        if self.backblend_mode in (BackBlendMode.PRELIMIT, BackBlendMode.BOTH):
            resharp = self._back_blend(resharp, matched)

        if self.sharp_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.TEMPORAL_PRESMOOTH):
            resharp = self._sharp_limit(resharp, bobbed)

        if self.backblend_mode in (BackBlendMode.POSTLIMIT, BackBlendMode.BOTH):
            resharp = self._back_blend(resharp, matched)

        return self._restore_noise(resharp, self.basic_noise_restore)

    def _final(self, basic: vs.VideoNode, bobbed: vs.VideoNode, denoised: vs.VideoNode) -> ConstantFormatVideoNode:

        smoothed = self.mv.degrain(basic, tr=self.final_tr, **self.final_degrain_args)

        masked = self._mask_shimmer(
            smoothed, bobbed, self.final_threshold, self.final_erosion_distance, self.final_over_dilation
        )

        if self.sharp_mode in (SharpLimitMode.SPATIAL_POSTSMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
            masked = self._sharp_limit(masked, bobbed)

        if self.lossless_mode == LosslessMode.POSTSMOOTH:
            masked = self._lossless(masked, denoised)

        return self._restore_noise(masked, self.final_noise_restore)

    def process(self) -> ConstantFormatVideoNode:

        tr = max(self.force_tr, self.denoise_tr, self.basic_tr, self.match_tr, self.final_tr)

        search, draft = self._prefilter(self.clip)

        self.mv = MVTools(draft, search, **self.preset)
        self.mv.analyze(tr=tr)

        denoised = self._denoise(draft, self.clip)

        if self.show_noise:
            return self.noise

        if self.input_type == InputType.REPAIR:
            denoised = reinterlace(denoised, self.tff)

        bobbed = self.basic_interpolator(denoised)

        if self.input_type == InputType.REPAIR:
            mask = self.mv.mask(search, direction=MVDirection.BACKWARD, kind=MaskMode.SAD)
            bobbed = denoised.std.MaskedMerge(bobbed, mask)

        basic_out = self._basic(bobbed, denoised)
        final_out = self._final(basic_out, bobbed, denoised)
        blurred = self._motion_blur(final_out, search)

        return blurred
