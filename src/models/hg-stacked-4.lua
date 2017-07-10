paths.dofile('layers/Residual.lua')

local function hourglass(n, numIn, numOut, inp)
    -- Upper branch
    local up1 = Residual(numIn,256)(inp)
    local up2 = Residual(256,256)(up1)
    local up4 = Residual(256,numOut)(up2)

    -- Lower branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numIn,256)(pool)
    local low2 = Residual(256,256)(low1)
    local low5 = Residual(256,256)(low2)
    local low6
    if n > 1 then
        low6 = hourglass(n-1,256,numOut,low5)
    else
        low6 = Residual(256,numOut)(low5)
    end
    local low7 = Residual(numOut,numOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({up4,up5})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, no stride, no padding
    local l_ = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l_))
end

function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,128)(r4)
    local r6 = Residual(128,256)(r5)

    -- First hourglass
    local hg1 = hourglass(4,256,512,r6)

    -- Linear layers to produce first set of predictions
    local l1 = lin(512,512,hg1)
    local l2 = lin(512,256,l1)

    -- First predicted heatmaps
    local out1 = nnlib.SpatialConvolution(256,outputDim[1][1],1,1,1,1,0,0)(l2)
    local out1_ = nnlib.SpatialConvolution(outputDim[1][1],256+128,1,1,1,1,0,0)(out1)

    -- Concatenate with previous linear features
    local cat1 = nn.JoinTable(2)({l2,pool})
    local cat1_ = nnlib.SpatialConvolution(256+128,256+128,1,1,1,1,0,0)(cat1)
    local int1 = nn.CAddTable()({cat1_,out1_})

    -- Second hourglass
    local hg2 = hourglass(4,256+128,512,int1)

    -- Linear layers to produce predictions again
    local l3 = lin(512,512,hg2)
    local l4 = lin(512,256,l3)

    -- Second predicted heatmaps
    local out2 = nnlib.SpatialConvolution(256,outputDim[2][1],1,1,1,1,0,0)(l4)
    local out2_ = nnlib.SpatialConvolution(outputDim[2][1],256+256,1,1,1,1,0,0)(out2)

    -- Concatenate with previous linear features
    local cat2 = nn.JoinTable(2)({l4,l2})
    local cat2_ = nnlib.SpatialConvolution(256+256,256+256,1,1,1,1,0,0)(cat2)
    local int2 = nn.CAddTable()({cat2_,out2_})

    -- Third hourglass
    local hg3 = hourglass(4,256+256,512,int2)

    -- Linear layers to produce predictions again
    local l5 = lin(512,512,hg3)
    local l6 = lin(512,256,l5)

    -- Third predicted heatmaps
    local out3 = nnlib.SpatialConvolution(256,outputDim[3][1],1,1,1,1,0,0)(l6)
    local out3_ = nnlib.SpatialConvolution(outputDim[3][1],256+256,1,1,1,1,0,0)(out3)

    -- Concatenate with previous linear features
    local cat3 = nn.JoinTable(2)({l6,l4})
    local cat3_ = nnlib.SpatialConvolution(256+256,256+256,1,1,1,1,0,0)(cat3)
    local int3 = nn.CAddTable()({cat3_,out3_})

    -- Fourth hourglass
    local hg4 = hourglass(4,256+256,512,int3)

    -- Linear layers to produce predictions again
    local l7 = lin(512,512,hg4)
    local l8 = lin(512,512,l7)

    -- Output heatmaps
    local out4 = nnlib.SpatialConvolution(512,outputDim[4][1],1,1,1,1,0,0)(l8)

    -- Final model
    local model = nn.gModule({inp}, {out1,out2,out3,out4})

    return model

end
