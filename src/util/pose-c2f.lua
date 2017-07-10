-- Get prediction coordinates
predDim = {nParts,3}

criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do criterion:add(nn.MSECriterion()) end

-- Code to generate training samples from raw images.
function generateSample(set, idx)
    local pts = annot[set]['part'][idx]
    local c = annot[set]['center'][idx]
    local s = annot[set]['scale'][idx]
    local z = annot[set]['zind'][idx]
    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])

    -- For single-person pose estimation with a centered/scaled figure
    local inp = crop(img, c, s, 0, opt.inputRes)
    local out = {}

    local sigma_2d = 2

    for i = 1,opt.nStack do
        local size_z = 2*torch.floor((6*sigma_2d*opt.resZ[i]/opt.outputRes+1)/2)+1
	local outTemp = torch.zeros(nParts*opt.resZ[i], opt.outputRes, opt.outputRes)
	for j = 1,nParts do
	    if pts[j][1] > 0 then -- Checks that there is a ground truth annotation
		drawGaussian3D(outTemp:sub((j-1)*opt.resZ[i]+1,j*opt.resZ[i]), transform(torch.add(pts[j],1), c, s, 0, opt.outputRes), torch.ceil(z[j]*opt.resZ[i]/opt.outputRes), sigma_2d, size_z)
            end
	end
	out[i] = outTemp
    end

    return inp,out
end

function preprocess(input, label)
    return input, label
end

function postprocess(set, idx, output)
    local preds = getPreds3D(output[#output])
    return preds
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8},h36m={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}}
    return heatmapAccuracy(output[#output],label[#label],nil,jntIdxs[opt.dataset])
end
