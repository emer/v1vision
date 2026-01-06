// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

//gosl:start

// Operations are the operations that can be performed.
type Operations int32 //enums:enum

const (
	NoOp Operations = iota

	// WrapPad wraps given padding width of float32 image around sides
	// i.e., padding for left side of image is the (mirrored) bits
	// from the right side of image, etc.
	// InImage -> OutImage, over InImageRGB (if 3, does all).
	WrapPad

	// FadePad wraps given padding width of float32 image around sides
	// i.e., padding for left side of image is the (mirrored) bits
	// from the right side of image, etc, and fades result toward average
	// edge value (passed in as arg).
	// InImage -> OutImage, over InImageRGB (if 3, does all).
	FadePad

	// LMSOpponents computes Long-Medium-Short (RGB) perceptually-based
	// color opponent values from InImage -> OutImage.
	// 0 = RedGreen (L-M), 1 = White-Black (grey), 2 = BlueYellow (S-(LM)),
	LMSOpponents

	// LMSComponents computes Long-Medium-Short (RGB) perceptually-based
	// color component values from InImage -> OutImage1, OutImage2.
	// For each image, the organization of components is designed to
	// align with the RGB components, using grey to fill in the extra bit.
	// Image1: 0 = Red (L), 1 = Green (M), 2 = Grey
	// Image2: 0 = Yellow (LM), 1 = Grey, 2 = Blue (S),
	LMSComponents

	// ConvolveImage applies a filter to Image, writing to Values.
	// InImage -> OutValue, using FilterType, FilterN
	ConvolveImage

	// ConvolveDiff applies two different filters to two different
	// [Image, component] inputs, computing their difference,
	// with positive values in 0 and negative values in 1 polarity,
	// at given feature dimension (innermost Values dimension).
	// This is used to compute e.g., on-center DoG to one color component
	// minus off-center to another component.
	ConvolveDiff

	// LogValues sets values to 1 + log of values * Gain.
	// InValue -> OutValue (can be the same).
	LogValues

	// MaxScalar computes Max over values.
	// InValue = values, OutScalar = result.
	MaxScalar

	// SumScalar computes Sum over values
	// InValue = values, OutScalar = result.
	SumScalar

	// MeanScalar computes Mean over values
	// InValue = values, OutScalar = result.
	MeanScalar

	// NormDiv normalizes values by scalar
	// InValue -> OutValue (can be same), InScalar = norm factor.
	NormDiv

	// NeighInhib4 computes neighbor inhibition, as an optional preliminary
	// step prior to KWTA. Currently only works with 4 angles (n features=4).
	// Each unit gets inhibition from same feature in nearest orthogonal neighbors.
	// Reduces redundancy of feature code.
	NeighInhib4

	// KWTAInhib computes k-winners-take-all inhibition, rate-code version,
	// based on overall levels of activity, over multiple iterations.
	KWTAInhib

	// MaxPool performs max-pooling over given pool size and spacing,
	// effectively reducing the dimensionality of the output by the
	// spacing factor. Size must = spacing or 2 * spacing.
	MaxPool

	// MaxPolarity performs max-pooling over the polarity (on vs. off)
	// dimension.
	MaxPolarity

	// MaxCopy performs simple max over 2 different values, for
	// aggregating different channels (e.g., colors) into a summary,
	// without changing the dimensionality.
	MaxCopy

	// LenSum4 performs V1 complex-cell length-summing, extending the
	// receptive field along the orientation angle one step.
	// Works on output from [MaxPolarity] (first polarity dimension),
	// only for the 4 angles case.
	LenSum4

	// EndStop4 performs V1 complex-cell end-stop, detecting an orthoginal
	// angle at the end of a length-sum line. Only for the 4 angles case.
	EndStop4

	// To4D copies from Values to Values4D for aggregating final results
	// across multiple feature dimensions (e.g., for assembling full V1 complex).
	To4D

	// MotionIntegrate does fast and slow motion integration from
	// values to values: InValue -> OutValue (should be different)
	MotionIntegrate

	// MotionStar computes starburst-style motion on integrated
	// fast and slow input values. Result is 4 * FilterN filter
	// outputs, for Left, Right, Down, Up motion directions.
	// InValue -> OutValue (different, X and Y are -1 in output).
	MotionStar

	// MotionFullField computes full-field summary of output from
	// MotionStar, into 4 Scalars for Left, Right, Down, Up.
	// Opposite directions compete.
	// OutScalar[0-3] = instantaneous full-field values per this frame
	// OutScalar[4-7] = integrated full-field values over time
	MotionFullField
)

// Op specifies an operation to perform.
// The full computational sequence is specified as a sequence of operations.
// This allows a full processing path to proceed with minimal transfers.
type Op struct {
	// Op is the operation to perform on this step
	Op Operations

	// NData is the number of data-parallel copies of everything to process
	// at once. Copied from V1Vision at op creation time.
	NData uint32

	// RunN is the total number of processors to deploy for this run
	// (i.e., the loop N for data parallel for loop, logically).
	// Actual run value will be * NData as well.
	RunN uint32

	// InImage is the index of an image to process as an input.
	InImage int32

	// InImageRGB is the RGB value to process of input image (0-2).
	// If 3, then all RGB are processed in one op (e.g., WrapPad)
	InImageRGB int32

	// InValue is the Values index input to use.
	InValue int32

	// InValue2 is the second Values index input to use, where needed.
	InValue2 int32

	// OutValue is the Values index output to write to.
	OutValue int32

	// OutValue4D is the Values4D index output to write to.
	OutValue4D int32

	// OutImage is the index of an image to send output for image ops.
	OutImage int32

	// OutImage2 is the index of a second image to send output for image ops.
	OutImage2 int32

	// FilterType is the type index of Filters to use.
	FilterType int32

	// FilterN is the number of filters within the FilterType to use.
	FilterN int32

	// FloatArg1 is a float argument -- e.g., used for gain multiplier
	// factor to apply.
	FloatArg1 float32

	// FloatArg2 is a float argument
	FloatArg2 float32

	// FloatArg3 is a float argument
	FloatArg3 float32

	// IntArg1 is an arbitrary integer arg, used for different ops.
	// e.g., PadWidth in WrapPad
	IntArg1 int32

	// InScalar is the Scalars index input to read from.
	InScalar int32

	// OutScalar is the Scalars index output to write to.
	OutScalar int32

	// Inhibs is the index of the Inhibs state variables to use.
	Inhibs int32

	// KWTA is the index of the KWTA parameters to use.
	KWTA int32

	pad, pad1, pad2 int32

	// Geom is the geometry to use for this operation.
	Geom Geom
}

// Run runs the operation on given run input index and NData index.
// (already range checked).
func (op *Op) Run(ri, ni int32) {
	switch op.Op {
	case ConvolveImage:
		op.ConvolveImage(ri, ni)
	case ConvolveDiff:
		op.ConvolveDiff(ri, ni)
	case WrapPad:
		op.WrapPad(ri, ni)
	case FadePad:
		op.FadePad(ri, ni)
	case LMSOpponents:
		op.LMSOpponents(ri, ni)
	case LMSComponents:
		op.LMSComponents(ri, ni)
	case LogValues:
		op.LogValues(ri, ni)
	case NormDiv:
		op.NormDiv(ri, ni)
	case NeighInhib4:
		op.NeighInhib4(ri, ni)
	case MaxPool:
		op.MaxPool(ri, ni)
	case MaxPolarity:
		op.MaxPolarity(ri, ni)
	case MaxCopy:
		op.MaxCopy(ri, ni)
	case LenSum4:
		op.LenSum4(ri, ni)
	case EndStop4:
		op.EndStop4(ri, ni)
	case To4D:
		op.To4D(ri, ni)
	case MotionIntegrate:
		op.MotionIntegrate(ri, ni)
	case MotionStar:
		op.MotionStar(ri, ni)
	default:
	}
}

func DoCurOp(i uint32) { //gosl:kernel
	op := GetCurOp(0)
	if i >= op.RunN*op.NData {
		return
	}
	ri := int32(i % op.RunN)
	ni := int32(i / op.RunN)
	op.Run(ri, ni)
}

//gosl:end

// RunOps runs all the operations.
func (vv *V1Vision) RunOps() {
	nops := len(vv.Ops)
	for i := range nops {
		vv.CurOp[0] = vv.Ops[i]
		ToGPU(CurOpVar)
		op := &vv.Ops[i]
		switch op.Op {
		case MaxScalar:
			RunMaxScalarX(int(op.RunN) * vv.NData)
			RunMaxScalarY(vv.NData)
		case SumScalar:
			RunSumScalarX(int(op.RunN) * vv.NData)
			RunSumScalarY(vv.NData)
		case MeanScalar:
			RunSumScalarX(int(op.RunN) * vv.NData)
			RunMeanScalarY(vv.NData)
		case KWTAInhib:
			kp := &vv.KWTAs[op.KWTA]
			RunKWTAInitLayer(vv.NData)
			RunKWTAInitPool(int(op.RunN) * vv.NData)
			for range kp.Iters {
				RunKWTAIterLayerX(int(op.Geom.Out.Y) * vv.NData)
				RunKWTAIterLayerY(vv.NData)
				RunKWTAIterPool(int(op.RunN) * vv.NData)
			}
		case MotionFullField:
			RunMotionFullFieldX(int(op.RunN) * vv.NData)
			RunMotionFullFieldY(2 * vv.NData)
		default:
			RunDoCurOp(int(op.RunN) * vv.NData)
		}
		if i < nops-1 {
			RunDone() // must wait to send next op
		}
	}
}
