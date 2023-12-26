## Fixed
object_name="alarm backpack bell blocks chicken cream elephant grandfather grandmother hat leather lion lunch_bag mario
 oil school_bus1 school_bus2 shoe shoe1 shoe2 shoe3 soap sofa sorter sorting_board stucking_cups teapot toaster train turtle"
NUM_OBJECTS=30

## Variable
SCALES1=2.0
SCALES2=1.0
NUM_SAMPLES=4

## Sample Images - Train NeuS - Export Mesh
for object in ${object_name}
do
  python generate.py --ckpt ckpt/syncdreamer-pretrain.ckpt --sample_num ${NUM_SAMPLES} --cfg_scales ${SCALES1} ${SCALES2} --decomposed_sampling --input gso-eval/${object}/000.png --output output_gso/${object} --elevation 30 --crop_size -1
  for id in $(seq ${NUM_SAMPLES})
  do
    python train_renderer.py -i output_gso/${object}/$((id-1)).png -n ${object}-$((id-1))-neus -b configs/neus.yaml -l output_gso_renderer
  done
done

## Eval PSNR, SSIM, LPIPS, E_flow, Chamfer Distance, Volume IoU - Calculate Statistics
for object in ${object_name}
do
  python eval_consistency.py --pr output_gso/${object} --gt gso-eval/${object} --name ${object} --num_samples ${NUM_SAMPLES}
  python eval_nvs.py --gt gso-eval/${object} --pr output_gso/${object} --name ${object} --num_samples ${NUM_SAMPLES}
  python eval_CD_score.py --pr output_gso/${object} --name ${object} --num_samples ${NUM_SAMPLES}
  for id in $(seq ${NUM_SAMPLES})
  do
    python eval_mesh.py --pr_mesh output_gso_renderer/${object}-$((id-1))-neus/mesh.ply --gt_mesh gso-eval/${object}/model.obj --gt_name ${object}
  done
done

python gso_nvs_statistics.py --log output_gso --num_samples ${NUM_SAMPLES} --num_objects ${NUM_OBJECTS}
python gso_mesh_statistics.py --log output_gso_renderer --num_samples ${NUM_SAMPLES} --num_objects ${NUM_OBJECTS}