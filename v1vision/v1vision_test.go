// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/v1std"
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
	assert.Equal(t, trg.Values, tsr.Values)
}

func TestDoGGrey(t *testing.T) {
	var vi v1std.DoGGrey
	var img v1std.Image

	filepath := "testdata/side-tee-128.png"

	vi.Defaults()
	vi.GPU = false
	img.Defaults()
	vi.Config(img.Size)
	im, _, err := imagex.Open(filepath)
	assert.NoError(t, err)
	vi.RunImage(&img, im)
	// fmt.Println(vi.Output)

	assertData(t, "DoGGrey", "Output", vi.Output)
}

func TestDoGColor(t *testing.T) {
	var vi v1std.DoGColor
	var img v1std.Image

	filepath := "testdata/macbeth.png"

	vi.Defaults()
	vi.GPU = false
	img.Defaults()
	vi.Config(img.Size)
	im, _, err := imagex.Open(filepath)
	assert.NoError(t, err)
	vi.RunImage(&img, im)
	// fmt.Println(vi.Output)

	assertData(t, "DoGColor", "Output", vi.Output)
}

func TestV1cGrey(t *testing.T) {
	var vi v1std.V1cGrey
	var img v1std.Image

	filepath := "testdata/side-tee-128.png"

	vi.Defaults()
	vi.GPU = false
	img.Defaults()
	vi.Config(img.Size)
	im, _, err := imagex.Open(filepath)
	assert.NoError(t, err)
	vi.RunImage(&img, im)
	// fmt.Println(vi.Output)

	assertData(t, "V1cGrey", "Output", vi.Output)
}

func TestV1cColor(t *testing.T) {
	var vi v1std.V1cColor
	var img v1std.Image

	filepath := "testdata/macbeth.png"

	vi.Defaults()
	vi.GPU = false
	img.Defaults()
	vi.Config(img.Size)
	im, _, err := imagex.Open(filepath)
	assert.NoError(t, err)
	vi.RunImage(&img, im)
	// fmt.Println(vi.Output)

	assertData(t, "V1cColor", "Output", vi.Output)
}
