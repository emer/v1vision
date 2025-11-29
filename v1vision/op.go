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

	// ConvolveImage applies a filter to Image, writing to Values.
	// InImage -> OutValue, using FilterType, FilterN
	ConvolveImage

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

	// NeighInhib computes neighbor inhibition, as an optional preliminary
	// step prior to KWTA. Currently only works with 4 angles (n features=4).
	NeighInhib

	// KWTAInhib computes k-winners-take-all inhibition, rate-code version,
	// based on overall levels of activity, over multiple iterations.
	KWTAInhib

	// MaxPool performs max-pooling over given pool size and spacing.
	// Size must = spacing or 2 * spacing.
	MaxPool

	// MotionIntegrate does fast and slow motion integration from
	// values to values: InValue -> OutValue (should be different)
	MotionIntegrate

	// MotionStar computes starburst-style motion on integrate
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

	// RunN is the total number of processors to deploy for this run
	// (i.e., the loop N for data parallel for loop, logically)
	RunN uint32

	// InImage is the index of an image to process as an input.
	InImage int32

	// InImageRGB is the RGB value to process of input image (0-2).
	// If 3, then all RGB are processed in one op (e.g., WrapPad)
	InImageRGB int32

	// InValue is the Values index input to use.
	InValue int32

	// OutValue is the Values index output to write to.
	OutValue int32

	// OutValue4D is the Values4D index output to write to.
	OutValue4D int32

	// OutImage is the index of an image to send output for image ops.
	OutImage int32

	// FilterType is the type index of Filters to use.
	FilterType int32

	// FilterN is the number of filters within the FilterType to use.
	FilterN int32

	// FloatArg1 is a float argument -- e.g., used for gain multiplier
	// factor to apply.
	FloatArg1 float32

	// FloatArg2 is a float argument
	FloatArg2 float32

	// IntArg1 is an arbitrary integer arg, used for different ops.
	// e.g., PadWidth in WrapPad
	IntArg1 int32

	// InScalar is the Scalars index input to read from.
	InScalar int32

	// OutScalar is the Scalars index output to write to.
	OutScalar int32

	// KWTA is the index of the KWTA parameters to use.
	KWTA int32

	// Geom is the geometry to use for this operation.
	Geom Geom
}

// Run runs the operation on given input index (already range checked).
func (op *Op) Run(i uint32) {
	switch op.Op {
	case ConvolveImage:
		op.ConvolveImage(i)
	case WrapPad:
		op.WrapPad(i)
	case LogValues:
		op.LogValues(i)
	case NormDiv:
		op.NormDiv(i)
	case NeighInhib:
		op.NeighInhib(i)
	case MaxPool:
		op.MaxPool(i)
	case MotionIntegrate:
		op.MotionIntegrate(i)
	case MotionStar:
		op.MotionStar(i)
	default:
	}
}

func DoCurOp(i uint32) { //gosl:kernel
	op := GetCurOp(0)
	if i >= op.RunN {
		return
	}
	op.Run(i)
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
			RunMaxScalarP1(int(op.RunN))
			RunMaxScalarP2(1)
		case SumScalar:
			RunSumScalarP1(int(op.RunN))
			RunSumScalarP2(1)
		case MeanScalar:
			RunSumScalarP1(int(op.RunN))
			RunMeanScalarP2(1)
		case KWTAInhib:
			kp := &vv.KWTAs[op.KWTA]
			RunKWTAInitLayer(1)
			RunKWTAInitPool(int(op.RunN))
			for range kp.Iters {
				RunKWTAIterLayer(1)
				RunKWTAIterPool(int(op.RunN))
			}
		case MotionFullField:
			RunMotionFullFieldP1(int(op.RunN))
			RunMotionFullFieldP2(2)
		default:
			RunDoCurOp(int(op.RunN))
		}
		if i < nops-1 {
			RunDone() // must wait to send next op
		}
	}
}
