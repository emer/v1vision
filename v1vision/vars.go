// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/lab/tensor"
)

//go:generate gosl -exclude=Update,Defaults,ShouldDisplay -max-buffer-size=2147483616

//gosl:start

// vars are all the global vars for axon GPU / CPU computation.
//
//gosl:vars
var (
	//////// Params

	// Ops are the sequence of operations to perform, called in order.
	//gosl:group Params
	//gosl:read-only
	Ops []Op

	//////// Filters

	// Filters are one general stack of rendered filters, sized to the max of each
	// of the inner dimensional values: [FilterTypes][FilterN][Y][X]
	// FilterTypes = different filter types (DoG, Gabor, etc)
	// FilterN = number of filters within the group (On, Off, angle, etc)
	// Y, X = sizes.
	//gosl:group Filters
	//gosl:read-only
	//gosl:dims 4
	Filters *tensor.Float32

	//////// Data

	// Images are float-valued image data: [ImageNo][RGB][Y][X],
	// sized to the max of each inner-dimensional value (RGB=3,
	// if more needed, use additional ImageNo)
	//gosl:group Data
	//gosl:dims 4
	Images *tensor.Float32

	// Values are intermediate input / output data: [ValueNo][Y][X][PosNeg][FilterN]
	// where FilterN corresponds to the different filters applied or other such data,
	// and PosNeg is 0 for positive (on) values and 1 for negative (off) values.
	//gosl:dims 5
	Values *tensor.Float32

	// Values4D are 4D aggregated data (e.g., outputs): [ValueNo][GpY][GpX][Y][X]
	//gosl:dims 5
	Values4D *tensor.Float32
)

//gosl:end
