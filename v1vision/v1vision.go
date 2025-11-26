// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"image"

	"cogentcore.org/lab/tensor"
)

//go:generate core generate -add-types -gosl

// V1Vision is a full set of filtering variables that is used for all
// v1vision processing, which can be installed onto the GPU.
type V1Vision struct {
	///////// Params

	// Ops are the sequence of operations to perform, called in order.
	Ops []Op

	//////// Filters

	// Filters are one general stack of rendered filters, sized to the max of each
	// of the inner dimensional values: [FilterTypes][FilterN][Y][X]
	// FilterTypes = different filter types (DoG, Gabor, etc)
	// FilterN = number of filters within the group (On, Off, angle, etc)
	// Y, X = sizes.
	Filters *tensor.Float32

	//////// Data

	// Images are float-valued image data: [ImageNo][RGB][Y][X],
	// sized to the max of each inner-dimensional value (RGB=3
	// if more needed, use additional ImageNo)
	Images *tensor.Float32

	// Values are intermediate input / output data: [ValueNo][Y][X][PosNeg][FilterN]
	// where FilterN corresponds to the different filters applied or other such data,
	// and PosNeg is 0 for positive (on) values and 1 for negative (off) values.
	Values *tensor.Float32

	// Values4D are 4D aggregated data (e.g., outputs): [ValueNo][GpY][GpX][Y][X]
	Values4D *tensor.Float32
}

// Init makes initial versions of all variables.
func (vv *V1Vision) Init() {
	vv.Ops = []Op{}
	vv.Filters = tensor.NewFloat32(0, 1, 1, 1)
	vv.Images = tensor.NewFloat32(0, 3, 1, 1)
	vv.Values = tensor.NewFloat32(0, 1, 1, 2, 1)
	vv.Values4D = tensor.NewFloat32(0, 1, 1, 1, 1)
}

// NewOp adds a new [Op]
func (vv *V1Vision) NewOp() *Op {
	n := len(vv.Ops)
	vv.Ops = append(vv.Ops, Op{})
	return &vv.Ops[n]
}

// NewImage adds a new image of given size. returns image index.
func (vv *V1Vision) NewImage(size image.Point) int {
	sizes := vv.Images.ShapeSizes()
	n := sizes[0]
	vv.Images.SetShapeSizes(n+1, 3, max(size.Y, sizes[2]), max(size.X, sizes[3]))
	return n
}

// NewValues adds a new Values of given sizes. returns value index.
func (vv *V1Vision) NewValues(y, x, filtN int) int {
	sizes := vv.Values.ShapeSizes()
	n := sizes[0]
	vv.Values.SetShapeSizes(n+1, max(y, sizes[1]), max(x, sizes[2]), 2, max(filtN, sizes[4]))
	return n
}

// NewValues4D adds a new Values4D of given sizes. returns value index.
func (vv *V1Vision) NewValues4D(gpY, gpX, y, x int) int {
	sizes := vv.Values4D.ShapeSizes()
	n := sizes[0]
	vv.Values4D.SetShapeSizes(n+1, max(gpY, sizes[1]), max(gpX, sizes[2]), max(y, sizes[3]), max(x, sizes[4]))
	return n
}

// NewFilter adds a new Filters of given sizes. returns filter index.
func (vv *V1Vision) NewFilter(filtN, y, x int) int {
	sizes := vv.Filters.ShapeSizes()
	n := sizes[0]
	vv.Filters.SetShapeSizes(n+1, max(filtN, sizes[1]), max(y, sizes[2]), max(x, sizes[3]))
	return n
}

// SetAsCurrent sets these as the current global values that are
// processed in the code (on the GPU).
func (vv *V1Vision) SetAsCurrent() {
	Ops = vv.Ops
	Filters = vv.Filters
	Images = vv.Images
	Values = vv.Values
	Values4D = vv.Values4D
}
