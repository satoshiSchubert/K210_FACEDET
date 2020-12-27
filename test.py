class ModelContainer():
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.patch_shape = (psize, psize, 3)
        self.batch_size_ = 4
        self._make_model_and_ops(None)

    def get_patch(self):
        patch = np.round((self._run(self.clipped_patch_)+1)*(255/2.0)).astype(np.uint8)
        patch *= RED_MASK
        patch[patch == 0] = 255
        return patch

    def assign_patch(self, new_patch):
        self._run(self.assign_patch_, {self.patch_placeholder_: new_patch})

    def reset_patch(self):
        self.assign_patch(np.zeros(self.patch_shape))
          
    def train_step(self, images, patch_transforms, second_stage_cls_labels, learning_rate=1.0,
                   dropout=None, rpn_nms_bboxes=None, rpn_nms_indices=None, patch_loss_weight=None):
        if (rpn_nms_bboxes is None) or \
           (rpn_nms_indices is None):
            rpn_nms_bboxes, rpn_nms_indices = self.inference_rpn(images, patch_transforms)

        feed_dict = { self.image_input_: images,
                      self.patch_transforms_: patch_transforms,
                      self.second_stage_cls_labels_: second_stage_cls_labels,
                      self.rpn_nms_bboxes_placeholder_: rpn_nms_bboxes,
                      self.rpn_nms_indices_placeholder_: rpn_nms_indices,
                      self.learning_rate_: learning_rate }
        
        if patch_loss_weight is not None:
            feed_dict[self.patch_loss_weight_] = patch_loss_weight
        
        tensors = [ self.train_op_,
                    self.loss_,
                    self.second_stage_cls_loss_, 
                    self.patch_loss_]

        train_op, loss, second_stage_cls_loss, patch_loss = self._run(tensors, feed_dict, dropout=dropout)

        return loss, second_stage_cls_loss, patch_loss
    
    def inference_rpn(self, images, patch_transforms):
        feed_dict = { self.image_input_: images,
                      self.patch_transforms_: patch_transforms }
        
        tensors = [self.rpn_nms_bboxes_,
                   self.rpn_nms_indices_ ]

        rpn_nms_bboxes, rpn_nms_indices = self._run(tensors, feed_dict)
        
        return rpn_nms_bboxes, rpn_nms_indices

    def inference(self, images, patch_transforms, rpn_nms_bboxes=None, rpn_nms_indices=None):
        if (rpn_nms_bboxes is None) or \
           (rpn_nms_indices is None):
            rpn_nms_bboxes, rpn_nms_indices = self.inference_rpn(images, patch_transforms)

        feed_dict = { self.image_input_: images,
                      self.patch_transforms_: patch_transforms,
                      self.rpn_nms_bboxes_placeholder_: rpn_nms_bboxes,
                      self.rpn_nms_indices_placeholder_: rpn_nms_indices }

        tensors = [ self.patched_input_,
                    self.second_stage_cls_scores_,
                    self.second_stage_loc_bboxes_ ]

        patched_imgs, second_stage_cls_scores, second_stage_loc_bboxes = self._run(tensors, feed_dict)
        patched_imgs = patched_imgs.astype(np.uint8)

        plot_detections(patched_imgs[0], scores=second_stage_cls_scores[0], bboxes=second_stage_loc_bboxes[0], min_threshold=0.2)
        
        return patched_imgs, second_stage_cls_scores, second_stage_loc_bboxes

    def _run(self, target, feed_dict=None, dropout=None):
        if feed_dict is None:
            feed_dict = {}
         
        if dropout is not None:
            feed_dict[self.dropout_] = dropout
    
        return self.sess.run(target, feed_dict=feed_dict)
    
    def _make_model_and_ops(self, patch_val):
        start = time.time()
        with self.sess.graph.as_default():
            tf.set_random_seed(1234)
            
            # Tensors are post-fixed with an underscore!
            self.image_input_ = tf.placeholder(tf.float32, shape=(None, psize, psize, 3), name='image_input')
            self.patch_transforms_ = tf.placeholder(tf.float32, shape=(None, 8), name='patch_transforms')

            patch_ = tf.get_variable('patch', self.patch_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
            self.patch_placeholder_ = tf.placeholder(dtype=tf.float32, shape=self.patch_shape, name='patch_placeholder')
            self.assign_patch_ = tf.assign(patch_, self.patch_placeholder_)
            self.clipped_patch_ = tf.tanh(patch_)

            self.dropout_ = tf.placeholder_with_default(1.0, [], name='dropout')
            patch_with_dropout_ = tf.nn.dropout(self.clipped_patch_, keep_prob=self.dropout_)
            patched_input_ = tf.clip_by_value(self._random_overlay(self.image_input_, patch_with_dropout_), clip_value_min=-1.0, clip_value_max=1.0)
            patched_input_ = tf.clip_by_value(tf.image.random_brightness(patched_input_, 10.0/255), -1.0, 1.0)
            self.patched_input_ = tf.fake_quant_with_min_max_vars((patched_input_ + 1)*127.5, min=0, max=255)

            # Create placeholders for NMS RPN inputs
            self.rpn_nms_bboxes_placeholder_ = tf.placeholder(tf.float32, shape=(None, 4), name='rpn_nms_bboxes')
            self.rpn_nms_indices_placeholder_ = tf.placeholder(tf.int32, shape=(None), name='rpn_nms_indices')

            detection_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                detection_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(detection_graph_def, name='detection',
                                    input_map={
                                               'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0':self.patched_input_,
                                               'Reshape:0':self.rpn_nms_bboxes_placeholder_,
                                               'Reshape_1:0':self.rpn_nms_indices_placeholder_,
                                              })

            # Recreate tensors we just replaced in the input_map
            self.rpn_nms_bboxes_ = tf.reshape(self.graph.get_tensor_by_name('detection/Reshape_3:0'), self.graph.get_tensor_by_name('detection/stack_3:0'), name='detection/Reshape')
            self.rpn_nms_indices_ = tf.reshape(self.graph.get_tensor_by_name('detection/ExpandDims_1:0'), self.graph.get_tensor_by_name('detection/Reshape_1/shape:0'), name='detection/Reshape_1')  

            # Patch Loss
            self.patch_loss_ = tf.nn.l2_loss(RED_MASK*(self.clipped_patch_ - np.tile(np.array([ 1.0, -0.9, -1]), (psize, psize, 1))))
            self.patch_loss_weight_ = tf.placeholder_with_default(1.0, [], 'patch_loss_weight')

            # Second-stage Class Loss
            self.second_stage_cls_scores_ = self.graph.get_tensor_by_name('detection/SecondStagePostprocessor/convert_scores:0')
            second_stage_cls_logits_ = self.graph.get_tensor_by_name('detection/SecondStagePostprocessor/scale_logits:0')
            self.second_stage_cls_labels_ = tf.placeholder(tf.float32, shape=second_stage_cls_logits_.shape, name='second_stage_cls_labels')
            second_stage_cls_losses_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(self.second_stage_cls_labels_, (-1, self.second_stage_cls_labels_.shape[2])),
                                                                                      logits=tf.reshape(second_stage_cls_logits_, (-1, second_stage_cls_logits_.shape[2]))) 
            second_stage_cls_losses_ = tf.reshape(second_stage_cls_losses_, (-1, self.second_stage_cls_labels_.shape[1]))
            second_stage_cls_losses_ = tf.divide(second_stage_cls_losses_, tf.to_float(self.second_stage_cls_labels_.shape[1]))
            self.second_stage_cls_loss_ = tf.reduce_sum(second_stage_cls_losses_)
           
            # Second-stage bounding boxes
            self.second_stage_loc_bboxes_ = self.graph.get_tensor_by_name('detection/SecondStagePostprocessor/Reshape_4:0')
    
            # Sum of weighted losses
            self.loss_ = self.patch_loss_*self.patch_loss_weight_ + (self.second_stage_cls_loss_)

            # Train our attack by only training on the patch variable
            self.learning_rate_ = tf.placeholder(tf.float32)
            self.train_op_ = tf.train.GradientDescentOptimizer(self.learning_rate_).minimize(self.loss_, var_list=[patch_])
            
            if patch_val is not None:
                self.assign_patch(patch_val)
            else:
                self.reset_patch()

            elapsed = time.time() - start
            print("Finished loading the model, took {:.0f}s".format(elapsed))
    

    def _random_overlay(self, imgs, patch):    
        red_mask = RED_MASK.astype(np.float32)
        white_mask = WHITE_MASK.astype(np.float32)
        
        red_mask = tf.stack([red_mask] * self.batch_size_)
        white_mask = tf.stack([white_mask] * self.batch_size_)
        padded_patch = tf.stack([patch] * self.batch_size_)
        
        white = tf.ones_like(red_mask) * 0.95
              
        red_mask = tf.contrib.image.transform(red_mask, self.patch_transforms_, 'BILINEAR')
        white_mask = tf.contrib.image.transform(white_mask, self.patch_transforms_, 'BILINEAR')
        padded_patch = tf.contrib.image.transform(padded_patch, self.patch_transforms_, 'BILINEAR')

        inverted_mask = (1 - red_mask - white_mask)

        return white * white_mask + imgs * inverted_mask + padded_patch * red_mask
    

    def _transform_vector(self, width, x_shift, y_shift, im_scale, rot_in_degrees):
        """
        If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
        then it maps the output point (x, y) to a transformed input point 
        (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
        where k = c0 x + c1 y + 1. 
        The transforms are inverted compared to the transform mapping input points to output points.
        """

        rot = float(rot_in_degrees) / 90. * (math.pi/2)

        # Standard rotation matrix
        # (use negative rot because tf.contrib.image.transform will do the inverse)
        rot_matrix = np.array(
            [[math.cos(-rot), -math.sin(-rot)],
            [math.sin(-rot), math.cos(-rot)]]
        )

        # Scale it
        # (use inverse scale because tf.contrib.image.transform will do the inverse)
        inv_scale = 1. / im_scale 
        xform_matrix = rot_matrix * inv_scale
        a0, a1 = xform_matrix[0]
        b0, b1 = xform_matrix[1]

        # At this point, the image will have been rotated around the top left corner,
        # rather than around the center of the image. 
        #
        # To fix this, we will see where the center of the image got sent by our transform,
        # and then undo that as part of the translation we apply.
        x_origin = float(width) / 2
        y_origin = float(width) / 2

        x_origin_shifted, y_origin_shifted = np.matmul(
            xform_matrix,
            np.array([x_origin, y_origin]),
        )

        x_origin_delta = x_origin - x_origin_shifted
        y_origin_delta = y_origin - y_origin_shifted

        # Combine our desired shifts with the rotation-induced undesirable shift
        a2 = x_origin_delta - (x_shift/(2*im_scale))
        b2 = y_origin_delta - (y_shift/(2*im_scale))

        # Return these values in the order that tf.contrib.image.transform expects
        return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

    def generate_random_transformation(self, scale_min=0.2, scale_max=0.6, width=psize, max_rotation=20):
        im_scale = np.random.uniform(low=scale_min, high=scale_max)

        padding_after_scaling = (1-im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

        rot = np.random.uniform(-max_rotation, max_rotation)

        return self._transform_vector(width, 
                                      x_shift=x_delta,
                                      y_shift=y_delta,
                                      im_scale=im_scale, 
                                      rot_in_degrees=rot)    

model = ModelContainer()
