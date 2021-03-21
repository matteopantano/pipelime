graph {

  dataset:image -> normalize:x;
  dataset:keypoints -> normalize_keypoints:x;
  normalize_keypoints:o -> keypoints_2_heatmap:x;
  keypoints_2_heatmap:o -> loss:x;
  normalize:o -> model:x;
  model:o -> loss:y;

}