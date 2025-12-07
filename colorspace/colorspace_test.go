// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/tolassert"
	"cogentcore.org/lab/tensor"
)

func assertData(t *testing.T, testName, tsrName string, tsr *tensor.Float32) {
	err := os.MkdirAll("testdata", 0777)
	assert.NoError(t, err)
	fn := filepath.Join("testdata", testName+"_"+tsrName+".tsv")
	save := imagex.UpdateTestImages
	exists, _ := fsx.FileExists(fn)
	if save || !exists {
		tensor.SaveCSV(tsr, fsx.Filename(fn), tensor.Tab)
		return
	}
	var trg tensor.Float32
	tensor.SetShapeFrom(&trg, tsr)
	err = tensor.OpenCSV(&trg, fsx.Filename(fn), tensor.Tab)
	assert.NoError(t, err)
	tolassert.EqualTolSlice(t, trg.Values, tsr.Values, 1.0e-7)
}

func TestOnMacbeth(t *testing.T) {
	clrs := MacbethFloats()
	n := clrs.DimSize(0)
	cmps := tensor.NewFloat32(n, 7)
	for i := range n {
		r := clrs.Value(i, 0)
		g := clrs.Value(i, 1)
		b := clrs.Value(i, 2)
		var lc, mc, sc, lmc, lvm, svlm, grey float32
		SRGBToLMSAll(r, g, b, &lc, &mc, &sc, &lmc, &lvm, &svlm, &grey)
		cmps.Set(lc, i, 0)
		cmps.Set(mc, i, 1)
		cmps.Set(sc, i, 2)
		cmps.Set(lmc, i, 3)
		cmps.Set(lvm, i, 4)
		cmps.Set(svlm, i, 5)
		cmps.Set(grey, i, 6)
	}
	// clrsz := clrs.Clone()
	// clrsz.SetShapeSizes(4, 6, 3)
	// cmpsz := cmps.Clone()
	// cmpsz.SetShapeSizes(4, 6, 7)
	// fmt.Println(clrsz)
	// fmt.Println(cmpsz)

	assertData(t, "Macbeth", "Output", cmps)
}
