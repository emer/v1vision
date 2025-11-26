// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

//gosl:start

// Operations are the operations that can be performed.
type Operations int32 //enums:enum

const (
	NoOp Operations = iota

	// ConvolveImage applies a filter to Image, writing to Values.
	ConvolveImage
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
	InImageRGB int32

	// InValue is the Values index input to use.
	InValue int32

	// OutValue is the Values index output to write to.
	OutValue int32

	// FilterType is the type index of Filters to use.
	FilterType int32

	// FilterN is the number of filters within the FilterType to use.
	FilterN int32

	// Gain is a multiplier factor to apply.
	Gain float32

	pad, pad1, pad2 int32

	// Geom is the geometry to use for this operation.
	Geom Geom
}

// Run runs the operation on given input index.
func (op *Op) Run(i uint32) {
	switch op.Op {
	case ConvolveImage:
		op.ConvolveImage(i)
	default:
	}
}

func Op0(i uint32) { //gosl:kernel
	op := GetOps(0)
	op.Run(i)
}

//gosl:end

// RunOps runs all the operations.
func (vv *V1Vision) RunOps() {
	for i := range vv.Ops {
		op := &vv.Ops[i]
		switch i {
		case 0:
			RunOp0(int(op.RunN))
		}
	}
}
