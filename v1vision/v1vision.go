// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/kwta"
)

//go:generate core generate -add-types -gosl

// V1Vision specifies a sequence of operations to perform on image
// input data, to simulate V1-level visual processing.
// The pipeline supports NData parallel data replications of everything.
type V1Vision struct {
	// NData is the number of data-parallel copies of everything to process
	// at once. Should be consistent throughout the stack. Copied into Ops
	// so it is available on the GPU.
	NData int

	// Ops are the sequence of operations to perform, called in order.
	Ops []Op

	// CurOp is the current operation to perform.
	CurOp []Op

	// KWTAs are KWTA inhibition parameters that can be used.
	KWTAs []kwta.KWTA

	// Filters are one general stack of rendered filters, sized to the max of each
	// of the inner dimensional values: [FilterTypes][FilterN][Y][X]
	// FilterTypes = different filter types (DoG, Gabor, etc)
	// FilterN = number of filters within the group (On, Off, angle, etc)
	// Y, X = sizes.
	Filters *tensor.Float32

	// Images are float-valued image data: [ImageNo][NData][RGB][Y][X],
	// sized to the max of each inner-dimensional value (RGB=3
	// if more needed, use additional ImageNo)
	Images *tensor.Float32

	// Values are intermediate input / output data:
	// [ValueNo][NData][Y][X][Polarity][FilterN]
	// where FilterN corresponds to the different filters applied or other such data,
	// and Polarity is 0 for positive (on) values and 1 for negative (off) values.
	Values *tensor.Float32

	// Values4D are 4D aggregated data (e.g., outputs):
	// [ValueNo][NData][PoolY][PoolX][UnitY][UnitX]
	Values4D *tensor.Float32

	// Scalars are scalar values for Sum, Max summary stats etc.
	// More efficient to use these versus using large Values allocations.
	// [values][NData]
	Scalars *tensor.Float32

	// Inhibs are [KWTAInhib] inhibitory state values:
	// [InhibNo][NData][PoolY][PoolX][InhibVarsN]
	Inhibs *tensor.Float32
}

// Init makes initial versions of all variables.
// Takes the number of data-parallel inputs to process in parallel.
func (vv *V1Vision) Init(ndata int) {
	vv.NData = max(1, ndata)
	vv.Ops = []Op{}
	vv.CurOp = make([]Op, 1)
	vv.Filters = tensor.NewFloat32(0, 1, 1, 1)
	vv.Images = tensor.NewFloat32(0, vv.NData, 3, 1, 1)
	vv.Values = tensor.NewFloat32(0, vv.NData, 1, 1, 2, 1)
	vv.Values4D = tensor.NewFloat32(0, vv.NData, 1, 1, 1, 1)
	vv.Scalars = tensor.NewFloat32(0, vv.NData)
	vv.Inhibs = tensor.NewFloat32(0, vv.NData, 1, 1, int(InhibVarsN))
}

// NewOp adds a new [Op]
func (vv *V1Vision) NewOp() *Op {
	n := len(vv.Ops)
	vv.Ops = append(vv.Ops, Op{NData: uint32(vv.NData)})
	return &vv.Ops[n]
}

// NewKWTAParams adds new [kwta.KWTA] params, initialized with defaults
func (vv *V1Vision) NewKWTAParams() *kwta.KWTA {
	n := len(vv.KWTAs)
	vv.KWTAs = append(vv.KWTAs, kwta.KWTA{})
	kv := &vv.KWTAs[n]
	kv.Defaults()
	return kv
}

// NewImage adds a new image of given size. returns image index.
func (vv *V1Vision) NewImage(size math32.Vector2i) int {
	sizes := vv.Images.ShapeSizes()
	n := sizes[0]
	vv.Images.SetShapeSizes(n+1, vv.NData, 3, max(int(size.Y), sizes[3]), max(int(size.X), sizes[4]))
	return n
}

// NewValues adds a new Values of given sizes. returns value index.
func (vv *V1Vision) NewValues(y, x, filtN int) int {
	sizes := vv.Values.ShapeSizes()
	n := sizes[0]
	vv.Values.SetShapeSizes(n+1, vv.NData, max(y, sizes[2]), max(x, sizes[3]), 2, max(filtN, sizes[5]))
	return n
}

// NewValues4D adds a new Values4D of given sizes. returns value index.
func (vv *V1Vision) NewValues4D(gpY, gpX, y, x int) int {
	sizes := vv.Values4D.ShapeSizes()
	n := sizes[0]
	vv.Values4D.SetShapeSizes(n+1, vv.NData, max(gpY, sizes[2]), max(gpX, sizes[3]), max(y, sizes[4]), max(x, sizes[5]))
	return n
}

// NewScalar adds given number of new Scalar(s), returning starting index.
func (vv *V1Vision) NewScalar(addN int) int {
	sizes := vv.Scalars.ShapeSizes()
	n := sizes[0]
	vv.Scalars.SetShapeSizes(n+addN, vv.NData)
	return n
}

// NewFilter adds a new Filters of given sizes. returns filter index.
// Note: if later adding filters of larger sizes, then initial filter data
// can be skewed, and you need to re-set it.
func (vv *V1Vision) NewFilter(filtN, y, x int) int {
	sizes := vv.Filters.ShapeSizes()
	n := sizes[0]
	vv.Filters.SetShapeSizes(n+1, max(filtN, sizes[1]), max(y, sizes[2]), max(x, sizes[3]))
	return n
}

// NewInhibs adds a new Inhibs of given pool sizes. returns index.
// Allocates 1 larger than pool size, as is actually needed.
func (vv *V1Vision) NewInhibs(py, px int) int {
	sizes := vv.Inhibs.ShapeSizes()
	n := sizes[0]
	vv.Inhibs.SetShapeSizes(n+1, vv.NData, max(py+1, sizes[2]), max(px+1, sizes[3]), int(InhibVarsN))
	return n
}

// SetAsCurrent sets these as the current global values that are
// processed in the code (on the GPU). If this was not the setter of
// the current variables, then the infrastructure variables are copied up
// to the GPU. It is thus best to have everything in one configuration to
// avoid switching costs.
func (vv *V1Vision) SetAsCurrent() {
	isCur := (Values == vv.Values)
	CurOp = vv.CurOp
	KWTAs = vv.KWTAs
	Filters = vv.Filters
	Images = vv.Images
	Values = vv.Values
	Values4D = vv.Values4D
	Scalars = vv.Scalars
	Inhibs = vv.Inhibs

	if GPUInitialized && !isCur {
		vv.ToGPUInfra()
	}
}

// GPUInit initializes the GPU and transfers Ops and Filters.
// Should have already called SetAsCurrent (needed for CPU and GPU).
func (vv *V1Vision) GPUInit() {
	GPUInit()
	UseGPU = true
	vv.ToGPUInfra()
}

// ToGPUInfra copies all the infrastructure for these filters up to
// the GPU. This is done in GPUInit, and
func (vv *V1Vision) ToGPUInfra() {
	ToGPUTensorStrides()
	ToGPU(CurOpVar, FiltersVar)
	if len(vv.KWTAs) > 0 {
		ToGPU(KWTAsVar)
	}
	// note: essential to copy up to GPU to init variable size.
	if vv.Values.Len() > 0 {
		ToGPU(ValuesVar)
	}
	if vv.Values4D.Len() > 0 {
		ToGPU(Values4DVar)
	}
	if vv.Scalars.Len() > 0 {
		ToGPU(ScalarsVar)
	}
	if vv.Inhibs.Len() > 0 {
		ToGPU(InhibsVar)
	}
}

func ImagesToGPU() {
	ToGPU(ImagesVar)
}

// Run transfers Images to GPU, does RunOps, retrieving the
// specified set of variables back from the GPU (if GPU running).
func (vv *V1Vision) Run(vars ...GPUVars) {
	ImagesToGPU()
	vv.RunOps()
	RunDone(vars...)
}

// ZeroValues sets all the values to zero.
// Useful when there are integrated accumulating values (e.g., motion).
func (vv *V1Vision) ZeroValues() {
	tensor.SetAllFloat64(vv.Values, 0)
	ToGPU(ValuesVar)
}
