// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

//gosl:start

// InhibVars are inhibition variables, stored in a tensor.Float32
// for FFFB inhibition computations.
type InhibVars int32 //enums:enum

const (
	// computed feedforward inhibition
	FFi InhibVars = iota

	// computed feedback inhibition (total)
	FBi

	// overall value of the inhibition. This is what is added into
	// the unit Gi inhibition level (along with any synaptic
	// unit-driven inhibition)
	Gi

	// original value of the inhibition (before pool or other effects)
	GiOrig

	// for pools, this is the layer-level inhibition that is MAX'd
	// with the pool-level inhibition to produce the net inhibition.
	LayGi

	// average Ge excitatory conductance values,
	// which drive FF inhibition
	GeAvg

	// max Ge excitatory conductance values,
	// which drive FF inhibition
	GeMax

	// average Act activation values,
	// which drive FB inhibition
	ActAvg

	// max Act activation values,
	// which drive FB inhibition
	ActMax
)

//gosl:end
