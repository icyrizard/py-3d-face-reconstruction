def reconstruct(model_shape_file, model_texture_file, image, asf_file):
    #assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    #assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'

    Vt_shape, s, n_shape_components, mean_value_points, triangles = pca.load(args.model_shape_file)
    Vt_texture, s_texture, n_texture_components, mean_values_texture, _ = pca.load(args.model_texture_file)

    InputPoints = imm.IMMPoints(filename=asf_file)
    input_image = InputPoints.get_image()

    MeanPoints = imm.IMMPoints(points_list=mean_value_points)
    MeanPoints.get_scaled_points(input_image.shape)

    while True:
        utils.reconstruct_texture(
            input_image,  # src image
            input_image,  # dst image
            Vt_texture,   # Vt
            InputPoints,  # shape points input
            MeanPoints,   # shape points mean
            mean_values_texture,  # mean texture
            triangles,  # triangles
            n_texture_components  # learned n_texture_components
        )

        dst = utils.get_texture(MeanPoints, mean_values_texture)

        cv2.imshow('original', InputPoints.get_image())
        cv2.imshow('reconstructed', input_image)
        cv2.imshow('main face', dst)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            break

    cv2.destroyAllWindows()
