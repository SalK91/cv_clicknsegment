
import cv2 as cv
crop_path='image_resize.png'


from array import array
import numpy as np


preds = [{'polys': [np.array([[0.23214285, 0.83928573],
       [0.26785713, 0.91071427],
       [0.30357143, 0.91071427],
       [0.30357143, 0.83928573],
       [0.3392857 , 0.83928573],
       [0.375     , 0.8035714 ],
       [0.4107143 , 0.8035714 ],
       [0.4107143 , 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.51785713, 0.625     ],
       [0.5535714 , 0.6964286 ],
       [0.58928573, 0.73214287],
       [0.625     , 0.76785713],
       [0.6964286 , 0.76785713],
       [0.6964286 , 0.73214287],
       [0.66071427, 0.6964286 ],
       [0.625     , 0.625     ],
       [0.58928573, 0.58928573],
       [0.58928573, 0.51785713],
       [0.625     , 0.51785713],
       [0.66071427, 0.48214287],
       [0.66071427, 0.3392857 ],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.51785713, 0.05357143],
       [0.44642857, 0.125     ],
       [0.44642857, 0.19642857],
       [0.4107143 , 0.26785713],
       [0.3392857 , 0.3392857 ],
       [0.23214285, 0.51785713]], dtype='float32')], 'scores': np.array([[0.7933912]], dtype='float32')}, {'polys': [np.array([[0.3392857 , 0.9464286 ],
       [0.4107143 , 0.9464286 ],
       [0.4107143 , 0.8035714 ],
       [0.44642857, 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.51785713, 0.625     ],
       [0.5535714 , 0.6964286 ],
       [0.58928573, 0.73214287],
       [0.625     , 0.76785713],
       [0.6964286 , 0.76785713],
       [0.66071427, 0.6964286 ],
       [0.625     , 0.66071427],
       [0.58928573, 0.58928573],
       [0.58928573, 0.51785713],
       [0.625     , 0.48214287],
       [0.66071427, 0.48214287],
       [0.66071427, 0.26785713],
       [0.6964286 , 0.19642857],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.5535714 , 0.05357143],
       [0.51785713, 0.08928572],
       [0.48214287, 0.08928572],
       [0.4107143 , 0.125     ],
       [0.4107143 , 0.16071428],
       [0.44642857, 0.19642857],
       [0.44642857, 0.23214285],
       [0.4107143 , 0.26785713],
       [0.26785713, 0.4107143 ],
       [0.23214285, 0.48214287],
       [0.23214285, 0.8035714 ],
       [0.26785713, 0.875     ],
       [0.30357143, 0.91071427]], dtype='float32')], 'scores': np.array([[0.7585983]], dtype='float32')}, {'polys': [np.array([[0.58928573, 0.5535714 ],
       [0.58928573, 0.51785713],
       [0.66071427, 0.51785713],
       [0.66071427, 0.26785713],
       [0.6964286 , 0.19642857],
       [0.6964286 , 0.125     ],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.5535714 , 0.05357143],
       [0.51785713, 0.08928572],
       [0.44642857, 0.08928572],
       [0.4107143 , 0.125     ],
       [0.4107143 , 0.16071428],
       [0.44642857, 0.19642857],
       [0.44642857, 0.23214285],
       [0.4107143 , 0.26785713],
       [0.4107143 , 0.30357143],
       [0.375     , 0.30357143],
       [0.3392857 , 0.3392857 ],
       [0.26785713, 0.4107143 ],
       [0.23214285, 0.48214287],
       [0.23214285, 0.83928573],
       [0.26785713, 0.875     ],
       [0.26785713, 0.91071427],
       [0.30357143, 0.91071427],
       [0.30357143, 0.83928573],
       [0.3392857 , 0.8035714 ],
       [0.4107143 , 0.8035714 ],
       [0.44642857, 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.5535714 , 0.6964286 ],
       [0.625     , 0.76785713],
       [0.6964286 , 0.76785713],
       [0.66071427, 0.6964286 ],
       [0.625     , 0.66071427]], dtype='float32')], 'scores': np.array([[0.749071]], dtype='float32')}, {'polys': [np.array([[0.4107143 , 0.26785713],
       [0.3392857 , 0.30357143],
       [0.26785713, 0.375     ],
       [0.23214285, 0.48214287],
       [0.23214285, 0.83928573],
       [0.26785713, 0.875     ],
       [0.26785713, 0.91071427],
       [0.30357143, 0.91071427],
       [0.30357143, 0.83928573],
       [0.3392857 , 0.8035714 ],
       [0.4107143 , 0.8035714 ],
       [0.4107143 , 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.51785713, 0.625     ],
       [0.5535714 , 0.6964286 ],
       [0.625     , 0.6964286 ],
       [0.625     , 0.76785713],
       [0.6964286 , 0.76785713],
       [0.6964286 , 0.73214287],
       [0.66071427, 0.6964286 ],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.51785713, 0.05357143],
       [0.44642857, 0.125     ],
       [0.4107143 , 0.16071428]], dtype='float32')], 'scores': np.array([[0.7390815]], dtype='float32')}, {'polys': [np.array([[0.30357143, 0.91071427],
       [0.3392857 , 0.9464286 ],
       [0.4107143 , 0.9464286 ],
       [0.4107143 , 0.8035714 ],
       [0.44642857, 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.51785713, 0.625     ],
       [0.5535714 , 0.6964286 ],
       [0.58928573, 0.73214287],
       [0.625     , 0.76785713],
       [0.66071427, 0.76785713],
       [0.66071427, 0.73214287],
       [0.625     , 0.6964286 ],
       [0.625     , 0.625     ],
       [0.58928573, 0.5535714 ],
       [0.58928573, 0.51785713],
       [0.625     , 0.51785713],
       [0.66071427, 0.48214287],
       [0.66071427, 0.3392857 ],
       [0.6964286 , 0.19642857],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.5535714 , 0.05357143],
       [0.51785713, 0.08928572],
       [0.48214287, 0.08928572],
       [0.4107143 , 0.125     ],
       [0.4107143 , 0.16071428],
       [0.44642857, 0.19642857],
       [0.44642857, 0.23214285],
       [0.4107143 , 0.26785713],
       [0.26785713, 0.4107143 ],
       [0.23214285, 0.48214287],
       [0.23214285, 0.8035714 ],
       [0.26785713, 0.875     ]], dtype='float32')], 'scores': np.array([[0.726653]], dtype='float32')}, {'polys': [np.array([[0.26785713, 0.875     ],
       [0.30357143, 0.91071427],
       [0.3392857 , 0.91071427],
       [0.3392857 , 0.83928573],
       [0.375     , 0.8035714 ],
       [0.4107143 , 0.8035714 ],
       [0.4107143 , 0.76785713],
       [0.44642857, 0.73214287],
       [0.48214287, 0.66071427],
       [0.51785713, 0.625     ],
       [0.5535714 , 0.6964286 ],
       [0.58928573, 0.73214287],
       [0.625     , 0.76785713],
       [0.66071427, 0.76785713],
       [0.66071427, 0.73214287],
       [0.625     , 0.6964286 ],
       [0.625     , 0.625     ],
       [0.58928573, 0.5535714 ],
       [0.58928573, 0.51785713],
       [0.625     , 0.51785713],
       [0.66071427, 0.48214287],
       [0.66071427, 0.3392857 ],
       [0.6964286 , 0.19642857],
       [0.66071427, 0.08928572],
       [0.625     , 0.08928572],
       [0.58928573, 0.05357143],
       [0.5535714 , 0.05357143],
       [0.51785713, 0.08928572],
       [0.48214287, 0.08928572],
       [0.44642857, 0.125     ],
       [0.4107143 , 0.125     ],
       [0.4107143 , 0.16071428],
       [0.44642857, 0.19642857],
       [0.44642857, 0.23214285],
       [0.4107143 , 0.26785713],
       [0.3392857 , 0.3392857 ],
       [0.26785713, 0.4107143 ],
       [0.23214285, 0.48214287],
       [0.23214285, 0.625     ],
       [0.19642857, 0.8035714 ]], dtype='float32')], 'scores': np.array([[0.69049704]], dtype='float32')}]

img = cv.imread(crop_path)

            
bestPoly = preds[0]['polys'][0]

# scale best poly coodrinates to size of image
# scale x-yaxis seperately
# unnest and then scale
print(bestPoly.shape)
#bestPoly = bestPoly.reshape((-1, 2))


bestPoly[:,0] = bestPoly[:,0] * img.shape[1]
bestPoly[:,1] = bestPoly[:,1] * img.shape[0]

#print(bestPoly)
#bestPoly = bestPoly.reshape((-1, 1, 2))
#bestPoly = bestPoly.tolist()
bestPoly = [np.array(bestPoly, dtype=np.int32)]

#print(bestPoly)
# add vertices to polygon in polylines function



cv.polylines(img, bestPoly, isClosed=True, color=(0, 255, 0), thickness=2)



#show image
cv.imshow('img', img)
cv.waitKey(0)
