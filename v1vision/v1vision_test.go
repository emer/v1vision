// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision_test

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"cogentcore.org/core/base/fsx"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/tolassert"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/edge"
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
	tolassert.EqualTolSlice(t, trg.Values, tsr.Values, 1.0e-5)
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

func TestMotionDoG(t *testing.T) {
	var vi v1std.MotionDoG
	imSize := image.Point{64, 64}
	vi.Defaults()
	vi.GPU = false
	vi.Config(imSize)

	imageTsr := vi.V1.Images.SubSpace(0).(*tensor.Float32)

	bar := image.Point{8, 16}
	velocity := math32.Vector2{1, 0}
	start := math32.Vector2{8, 8}

	vi.Motion.NormInteg = 0
	pos := start
	for range 16 {
		pad := vi.Geom.Border.V()
		tensor.SetAllFloat64(imageTsr, 0)
		for y := range bar.Y {
			py := int(math32.Round(pos.Y))
			yp, _ := edge.Edge(y+py, imSize.Y, true)
			for x := range bar.X {
				px := int(math32.Round(pos.X))
				xp, _ := edge.Edge(x+px, imSize.X, true)
				imageTsr.Set(1, 0, int(pad.Y)+yp, int(pad.X)+xp)
			}
		}
		pos = pos.Add(velocity)
		vi.Run()
	}

	assertData(t, "MotionDoG", "FullField", &vi.FullField)
}
