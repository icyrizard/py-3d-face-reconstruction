def eigenCallback(x):
    global reconstructed

    imm_orig = IMMPoints(filename=args.asf[index])
    img = cv2.imread('data/imm_face_db/' + imm_orig.filename)

    yk = np.dot(Vt[:n_components], X[index].T)
    reconstructed = (np.dot(Vt[:n_components].T, yk) + mean_values).reshape((58, 2))

    imm_orig.show_on_img(img)
    cv2.imshow('image', img)

