// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1vision"
)

// V1cParams has the parameters for a given size of V1c.
type V1cParams struct {
	// Name is the name of this size.
	Name string

	// V1 simple gabor filter parameters.
	V1sGabor gabor.Filter

	// Zoom is the zoom factor: divides effective image size in setting params.
	Zoom float32

	// geometry of input, output for V1 simple-cell processing.
	V1sGeom v1vision.Geom `edit:"-"`

	// geometry of input, output for V1 complex-cell processing from V1s inputs.
	V1cGeom v1vision.Geom `edit:"-"`

	// Output contains this 4D filter output, in correct shape.
	Output tensor.Float32

	// Values4D index of output.
	OutIdx int

	gaborIdx int
}

// Config configures geometry and filter sizes. The border is used directly, and
// should be consistent within each over field-of-view size.
func (vp *V1cParams) Config(nm string, zoom float32, border, v1sSize, v1sSpace int) *V1cParams {
	vp.Name = nm
	vp.Zoom = zoom
	vp.V1sGabor.Defaults()
	vp.V1sGabor.SetSize(v1sSize, v1sSpace)
	vp.V1sGeom.Set(math32.Vec2i(border, border), math32.Vec2i(v1sSpace, v1sSpace), math32.Vec2i(v1sSize, v1sSize))
	return vp
}

// SetImageSize sets the image size accordingly, dividing by Zoom factor.
func (vp *V1cParams) SetImageSize(imageSize image.Point) {
	isz := math32.FromPoint(imageSize).DivScalar(vp.Zoom).ToPointRound()
	vp.V1sGeom.SetImageSize(isz)
}

func (vp *V1cParams) V1Config(vi *V1cMulti, lms, kwtaIdx int) {
	nang := vp.V1sGabor.NAngles
	// V1s simple
	ftyp := vi.V1.NewFilter(nang, vp.V1sGabor.Size, vp.V1sGabor.Size)
	vp.gaborIdx = ftyp
	vi.V1.GaborToFilter(ftyp, &vp.V1sGabor)
	inh := vi.V1.NewInhibs(int(vp.V1sGeom.Out.Y), int(vp.V1sGeom.Out.X))
	lmsMap := [3]int{1, int(v1vision.RedGreen), int(v1vision.BlueYellow)}
	var v1sIdxs [3]int
	for irgb := range 3 {
		out := vi.V1.NewConvolveImage(lms, lmsMap[irgb], ftyp, nang, vp.V1sGabor.Gain, &vp.V1sGeom)
		v1out := out
		if vi.V1sKWTA.On.IsTrue() {
			ninh := 0
			if vi.V1sNeighInhib.On {
				ninh = vi.V1.NewNeighInhib4(out, nang, vi.V1sNeighInhib.Gi, &vp.V1sGeom)
			}
			v1out = vi.V1.NewKWTA(out, ninh, nang, kwtaIdx, inh, &vp.V1sGeom)
		}
		v1sIdxs[irgb] = v1out
	}
	mcout := vi.V1.NewValues(int(vp.V1sGeom.Out.Y), int(vp.V1sGeom.Out.X), nang)
	vi.V1.NewMaxCopy(v1sIdxs[0], v1sIdxs[1], mcout, nang, &vp.V1sGeom)
	vi.V1.NewMaxCopy(v1sIdxs[2], mcout, mcout, nang, &vp.V1sGeom)

	// V1c complex
	vp.V1cGeom.SetFilter(math32.Vec2i(0, 0), math32.Vec2i(2, 2), math32.Vec2i(2, 2), vp.V1sGeom.Out.V())
	mpout := vi.V1.NewMaxPolarity(mcout, nang, &vp.V1sGeom)
	pmpout := vi.V1.NewMaxPool(mpout, 1, nang, &vp.V1cGeom)
	lsout := vi.V1.NewLenSum4(pmpout, nang, &vp.V1cGeom)
	esout := vi.V1.NewEndStop4(pmpout, lsout, nang, &vp.V1cGeom)

	// To4D
	out4Rows := vi.Out4Rows()
	out4 := vi.V1.NewValues4D(int(vp.V1cGeom.Out.Y), int(vp.V1cGeom.Out.X), out4Rows, nang)
	vp.OutIdx = out4
	vp.Output.SetShapeSizes(int(vp.V1cGeom.Out.Y), int(vp.V1cGeom.Out.X), out4Rows, nang)

	vi.V1.NewTo4D(lsout, out4, 1, nang, 0, &vp.V1cGeom)
	vi.V1.NewTo4D(esout, out4, 2, nang, 1, &vp.V1cGeom)
	if vi.SplitColor {
		poutg := vi.V1.NewMaxPool(v1sIdxs[0], 2, nang, &vp.V1cGeom)
		poutrg := vi.V1.NewMaxPool(v1sIdxs[1], 2, nang, &vp.V1cGeom)
		poutby := vi.V1.NewMaxPool(v1sIdxs[2], 2, nang, &vp.V1cGeom)

		vi.V1.NewTo4D(poutg, out4, 2, nang, 3, &vp.V1cGeom)
		vi.V1.NewTo4D(poutrg, out4, 2, nang, 5, &vp.V1cGeom)
		vi.V1.NewTo4D(poutby, out4, 2, nang, 7, &vp.V1cGeom)
	} else {
		pout := vi.V1.NewMaxPool(mcout, 2, nang, &vp.V1cGeom)
		vi.V1.NewTo4D(pout, out4, 2, nang, 3, &vp.V1cGeom)
	}
}

func (vp *V1cParams) UpdateFilter(vi *V1cMulti) {
	vi.V1.GaborToFilter(vp.gaborIdx, &vp.V1sGabor)
}

// SetOutput sets the output for this filter.
func (vp *V1cParams) SetOutput(vi *V1cMulti) {
	out := vi.V1.Values4D.SubSpace(vp.OutIdx).(*tensor.Float32)
	tensor.CopyFromLargerShape(&vp.Output, out)
}

//////// DoG Color

// DoGColorParams has the parameters for a given size of DoG color.
type DoGColorParams struct {
	// Name is the name of this size.
	Name string

	// DoG color filter parameters. Generally have larger fields,
	// and no spatial tuning (i.e., OnSigma == OffSigma), consistent
	// with blob cells.
	DoG dog.Filter

	// Zoom is the zoom factor: divides effective image size in setting params.
	Zoom float32

	// geometry of DoG color contrast outputs.
	Geom v1vision.Geom `edit:"-"`

	// Output contains this 4D filter output, in correct shape.
	Output tensor.Float32

	// Values4D indexes of output.
	OutIdx int

	dogIdx int
}

// Config configures geometry and filter sizes. The border is used directly, and
// should be consistent within each over field-of-view size.
func (vi *DoGColorParams) Config(nm string, zoom float32, border, dogSize int) *DoGColorParams {
	vi.Name = nm
	vi.Zoom = zoom
	vi.DoG.Spacing = dogSize
	vi.DoG.Size = dogSize
	vi.DoG.Gain = 8          // color channels are weaker than grey
	vi.DoG.OnGain = 1        // balanced
	vi.DoG.SetSameSigma(0.5) // no spatial component, just pure contrast
	vi.Geom.Set(math32.Vec2i(border, border), math32.Vec2i(dogSize, dogSize), math32.Vec2i(dogSize, dogSize))
	return vi
}

// SetImageSize sets the image size accordingly, dividing by Zoom factor.
func (vp *DoGColorParams) SetImageSize(imageSize image.Point) {
	isz := math32.FromPoint(imageSize).DivScalar(vp.Zoom).ToPointRound()
	vp.Geom.SetImageSize(isz)
}

func (vp *DoGColorParams) V1Config(vi *V1cMulti, lmsRG, lmsBY, kwtaIdx int) {
	out := vi.V1.NewValues(int(vp.Geom.Out.Y), int(vp.Geom.Out.X), 2)
	dogFt := vi.V1.NewDoGOnOff(&vp.DoG, &vp.Geom)
	vp.dogIdx = dogFt

	vi.V1.NewConvolveDiff(lmsRG, v1vision.Red, lmsRG, v1vision.Green, dogFt, 0, 1, out, 0, 1, vp.DoG.OnGain, &vp.Geom)
	vi.V1.NewConvolveDiff(lmsBY, v1vision.Blue, lmsBY, v1vision.Yellow, dogFt, 0, 1, out, 1, 1, vp.DoG.OnGain, &vp.Geom)

	if vi.DoGKWTA.On.IsTrue() {
		inh := vi.V1.NewInhibs(int(vp.Geom.Out.Y), int(vp.Geom.Out.X))
		out = vi.V1.NewKWTA(out, 0, 2, kwtaIdx, inh, &vp.Geom)
	}

	// To4D
	out4 := vi.V1.NewValues4D(int(vp.Geom.Out.Y), int(vp.Geom.Out.X), 2, 2)
	vp.OutIdx = out4
	vp.Output.SetShapeSizes(int(vp.Geom.Out.Y), int(vp.Geom.Out.X), 2, 2)
	vi.V1.NewTo4D(out, out4, 2, 2, 0, &vp.Geom)
}

func (vp *DoGColorParams) UpdateFilter(vi *V1cMulti) {
	vi.V1.DoGOnOffToFilter(vp.dogIdx, &vp.DoG)
}

// SetOutput sets the output for this filter.
func (vp *DoGColorParams) SetOutput(vi *V1cMulti) {
	out := vi.V1.Values4D.SubSpace(vp.OutIdx).(*tensor.Float32)
	tensor.CopyFromLargerShape(&vp.Output, out)
}

//////// V1cMulti

// V1cMulti does color V1 complex (V1c) filtering and DoG color filtering
// across multiple different resolutions and filter sizes.
// V1c starts with simple cells (V1s) and adds length sum and end stopping.
// KWTA inhibition operates on the V1s step. DoG does Red-Green and Blue-Yellow
// color contrasts, capturing the chromatic response properties of color blob cells.
// Call Defaults and then set any custom params, then call Config.
// Results are in Output tensor after Run(), which has a 4D shape.
type V1cMulti struct {
	// GPU means use the GPU by default (does GPU initialization) in Config.
	// To change what is actually used at the moment of running,
	// set [v1vision.UseGPU].
	GPU bool

	// SplitColor records separate rows in V1c simple summary for each color.
	// Otherwise records the max across all colors.
	SplitColor bool

	// ColorGain is an extra gain for color channels,
	// which are lower contrast in general.
	ColorGain float32 `default:"8"`

	// V1sNeighInhib specifies neighborhood inhibition for V1s.
	// Each unit gets inhibition from same feature in nearest orthogonal
	// neighbors. Reduces redundancy of feature code.
	V1sNeighInhib kwta.NeighInhib

	// V1sKWTA has the kwta inhibition parameters for V1s.
	V1sKWTA kwta.KWTA

	// DoGKWTA has the kwta inhibition parameters for DoG Color blobs.
	DoGKWTA kwta.KWTA

	// V1cParams has the configured geometries for different V1c sizes.
	V1cParams []*V1cParams

	// DoGParams has the configured geometries for different DoG color
	// sizes.
	DoGParams []*DoGColorParams

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// Image manages images.
	Image Image

	fadeOpIdx int
}

func (vi *V1cMulti) Defaults() {
	vi.GPU = true
	vi.ColorGain = 8
	vi.SplitColor = true
	vi.Image.Defaults()
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	vi.DoGKWTA.Defaults()
	vi.DoGKWTA.Layer.On.SetBool(false) // non-spatial, mainly for differentiation within pools
	vi.DoGKWTA.Pool.Gi = 1.2
}

func (vi *V1cMulti) AddV1cParams() *V1cParams {
	gm := &V1cParams{}
	vi.V1cParams = append(vi.V1cParams, gm)
	return gm
}

func (vi *V1cMulti) AddDoGParams() *DoGColorParams {
	gm := &DoGColorParams{}
	vi.DoGParams = append(vi.DoGParams, gm)
	return gm
}

// StdLowMed16DegZoom1 configures a standard 16 degree parafovial
// field of view (FOV), with Low and Medium resolution V1c filters
// and 1 level of spatial zoom (8 degrees),
// Along with corresponding low and medium resolution color DoGs.
// This operates on 128x128 image content.
func (vi *V1cMulti) StdLowMed16DegZoom1() {
	vi.Image.Size = image.Point{128, 128}
	// target full wrap/pad image size = 128 + 12 * 2 = 152
	vi.AddV1cParams().Config("L16", 1, 12, 24, 8) // 128 / 8 = 16
	vi.AddV1cParams().Config("M16", 1, 12, 12, 4) // 128 / 4 = 32
	// 	vi.AddV1cParams().Config("H16", 1, 12, 6, 2) // not used in LVis small

	// 64 + 44*2 = 152
	vi.AddV1cParams().Config("M8", 2, 44, 12, 4)
	vi.AddV1cParams().Config("H8", 2, 44, 6, 2)

	vi.AddDoGParams().Config("L16", 1, 12, 16)
	vi.AddDoGParams().Config("M16", 1, 12, 8)

	vi.AddDoGParams().Config("L8", 2, 44, 8)
	vi.AddDoGParams().Config("M8", 2, 44, 4)
}

// StdLowMed16DegNoDoG configures a standard 16 degree parafovial
// field of view (FOV), with Low and Medium resolution V1c filters.
// This operates on 128x128 image content.
func (vi *V1cMulti) StdLowMed16DegNoDoG() {
	vi.Image.Size = image.Point{128, 128}
	// target full wrap/pad image size = 128 + 12 * 2 = 152
	vi.AddV1cParams().Config("L16", 1, 12, 24, 8) // 128 / 8 = 16
	vi.AddV1cParams().Config("M16", 1, 12, 12, 4) // 128 / 4 = 32
}

func (vi *V1cMulti) Out4Rows() int {
	out4Rows := 5
	if vi.SplitColor {
		out4Rows = 9
	}
	return out4Rows
}

// Config configures the filtering pipeline with all the current parameters.
func (vi *V1cMulti) Config() {
	for _, vp := range vi.V1cParams {
		vp.SetImageSize(vi.Image.Size)
	}
	for _, vp := range vi.DoGParams {
		vp.SetImageSize(vi.Image.Size)
	}
	v1sGeom := &vi.V1cParams[0].V1sGeom
	inSz := v1sGeom.In.V()

	vi.V1.Init()
	*vi.V1.NewKWTAParams() = vi.V1sKWTA
	v1sKwtaIdx := 0
	*vi.V1.NewKWTAParams() = vi.DoGKWTA
	dogKwtaIdx := 1
	img := vi.V1.NewImage(inSz)
	wrap := vi.V1.NewImage(inSz)
	lmsOp := vi.V1.NewImage(inSz)
	lmsRG := vi.V1.NewImage(inSz)
	lmsBY := vi.V1.NewImage(inSz)
	_, _ = lmsRG, lmsBY

	vi.fadeOpIdx = vi.V1.NewFadeImage(img, 3, wrap, int(v1sGeom.Border.X), .5, .5, .5, v1sGeom)
	vi.V1.NewLMSOpponents(wrap, lmsOp, vi.ColorGain, v1sGeom)
	if len(vi.DoGParams) > 0 {
		dogGeom := &vi.DoGParams[0].Geom
		vi.V1.NewLMSComponents(wrap, lmsRG, lmsBY, vi.ColorGain, dogGeom)
	}

	for _, vp := range vi.V1cParams {
		vp.V1Config(vi, lmsOp, v1sKwtaIdx)
	}
	for _, vp := range vi.DoGParams {
		vp.V1Config(vi, lmsRG, lmsBY, dogKwtaIdx)
	}

	// critical to go back and fix all the filters.
	for _, vp := range vi.V1cParams {
		vp.UpdateFilter(vi)
	}
	for _, vp := range vi.DoGParams {
		vp.UpdateFilter(vi)
	}

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// RunImage runs the configured filtering pipeline.
// on given Image, using given [Image] handler.
func (vi *V1cMulti) RunImage(img image.Image) {
	vi.V1.SetAsCurrent()
	v1vision.UseGPU = vi.GPU
	v1sGeom := &vi.V1cParams[0].V1sGeom
	vi.Image.SetImageRGB(&vi.V1, img, int(v1sGeom.Border.X))
	r, g, b := v1vision.EdgeAvg(vi.Image.Tsr, int(v1sGeom.Border.X))
	vi.V1.SetFadeRGB(vi.fadeOpIdx, r, g, b)
	vi.V1.Run(v1vision.Values4DVar)
	for _, vp := range vi.V1cParams {
		vp.SetOutput(vi)
	}
	for _, vp := range vi.DoGParams {
		vp.SetOutput(vi)
	}
}
