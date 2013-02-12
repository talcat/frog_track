# trackbyhand.py
#
# Final script that allows a user to draw a box aroun
# the ROI or point (care about the center of the box) in each frame.


from sel_prim_roi import * #contains select_prim_ROI function
    


if __name__ == "__main__":
    im = imread('im-0.png')
    test = select_prim_ROI(im)
    (minr, maxr), (minc, maxc) = test
    imshow(im[minr:maxr, minc:maxc, :])
    show()
    print test

