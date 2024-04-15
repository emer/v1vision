// Code generated by "core generate -add-types"; DO NOT EDIT.

package colorspace

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "github.com/emer/vision/v2/colorspace.LMSComponents", IDName: "lms-components", Doc: "LMSComponents are different components of the LMS space\nincluding opponent contrasts and grey"})

var _ = types.AddType(&types.Type{Name: "github.com/emer/vision/v2/colorspace.Opponents", IDName: "opponents", Doc: "Opponents enumerates the three primary opponency channels:\nWhiteBlack, RedGreen, BlueYellow\nusing colloquial \"everyday\" terms."})

var _ = types.AddType(&types.Type{Name: "github.com/emer/vision/v2/colorspace.SRGBToOp", IDName: "srgb-to-op", Doc: "SRGBToOp implements a lookup-table for the conversion of\nSRGB components to LMS color opponent values.\nAfter all this, it looks like the direct computation is faster\nthan the lookup table!  In any case, it is all here and reasonably\naccurate (mostly under 1.0e-4 according to testing)", Fields: []types.Field{{Name: "Levels", Doc: "number of levels in the lookup table -- linear interpolation used"}, {Name: "Table", Doc: "lookup table"}}})