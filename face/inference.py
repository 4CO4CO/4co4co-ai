import cv2
import numpy as np 
import os 

import torch 
from torchvision import transforms

from emotic import Emotic 


def process_images(context_norm, body_norm, image_context_path=None, image_context=None, image_body=None, bbox=None):
  if image_context is None and image_context_path is None:
    raise ValueError('both image_context and image_context_path cannot be none. Please specify one of the two.')
  if image_body is None and bbox is None: 
    raise ValueError('both body image and bounding box cannot be none. Please specify one of the two')

  if image_context_path is not None:
    image_context =  cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
  
  if bbox is not None:
    image_body = image_context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
  
  image_context = cv2.resize(image_context, (224,224))
  image_body = cv2.resize(image_body, (128,128))
  
  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  context_norm = transforms.Normalize(context_norm[0], context_norm[1])  
  body_norm = transforms.Normalize(body_norm[0], body_norm[1])

  image_context = context_norm(test_transform(image_context)).unsqueeze(0)
  image_body = body_norm(test_transform(image_body)).unsqueeze(0)

  return image_context, image_body  


def infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models,
          image_context_path=None, image_context=None, image_body=None, bbox=None, to_print=True):

  image_context, image_body = process_images(
      context_norm, body_norm,
      image_context_path=image_context_path,
      image_context=image_context,
      image_body=image_body,
      bbox=bbox
  )

  model_context, model_body, emotic_model = models

  with torch.no_grad():
    image_context = image_context.to(device)
    image_body = image_body.to(device)
    
    pred_context = model_context(image_context)
    pred_body = model_body(image_body)
    pred_cat, pred_cont = emotic_model(pred_context, pred_body)  # pred_cat: (1, C), pred_cont: (1, 3)
    pred_cat = pred_cat.squeeze(0).float().cpu()
    pred_cont = pred_cont.squeeze(0).cpu().numpy()  # (3,)

    # pred_cat이 확률인지 로짓인지 자동 판별 후 확률로 변환
    # (일반적으로 로짓이면 값 범위가 [-inf, +inf], 확률이면 [0,1])
    if pred_cat.min() < 0.0 or pred_cat.max() > 1.0:
      probs = torch.sigmoid(pred_cat)
    else:
      probs = pred_cat  # 이미 확률

    # thresholds도 확률 기준이라고 가정 (일반적인 F1 튜닝 산출물)
    # thresholds shape: (C,)
    if thresholds.is_cuda:
      thresholds = thresholds.cpu()
    bool_cat_pred = torch.gt(probs, thresholds)

  # 선택된 감정/확률
  cat_emotions = []
  selected_with_probs = []
  for i in range(len(bool_cat_pred)):
    if bool_cat_pred[i]:
      emotion = ind2cat[i]
      p = probs[i].item()
      cat_emotions.append(emotion)
      selected_with_probs.append((emotion, p))

  if to_print:
    print('\n Image predictions')
    print('Continuous Dimnesions Predictions') 
    for i in range(len(pred_cont)):
      print('Continuous %10s %.5f' %(ind2vad[i], 10*pred_cont[i]))
    print('Categorical Emotion Predictions (selected by threshold)')
    for emotion, p in selected_with_probs:
      print(f'Categorical {emotion:16s}  Prob={p:.4f}')

  # 확률 딕셔너리
  probs_dict = {ind2cat[i]: float(probs[i].item()) for i in range(len(probs))}
  return cat_emotions, 10*pred_cont, probs_dict, selected_with_probs


def inference_emotic(images_list, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args):

  with open(images_list, 'r') as f:
    lines = f.readlines()
  
  device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'model_context1.pth'), map_location=device).to(device)
  model_body = torch.load(os.path.join(model_path,'model_body1.pth'), map_location=device).to(device)
  emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth'), map_location=device).to(device)
  model_context.eval()
  model_body.eval()
  emotic_model.eval()
  models = [model_context, model_body, emotic_model]


  result_file = os.path.join(result_path, 'inference_list.txt')
  result_file_probs = os.path.join(result_path, 'inference_list_with_probs.txt')
  for fp in [result_file, result_file_probs]:
    with open(fp, 'w') as f:
      pass
  
  for idx, line in enumerate(lines):
    image_context_path, x1, y1, x2, y2 = line.split('\n')[0].split(' ')
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    pred_cat, pred_cont, probs_dict, selected_with_probs = infer(
        context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models,
        image_context_path=image_context_path, bbox=bbox)

    write_line = []
    write_line.append(image_context_path)
    for emotion in pred_cat:
      write_line.append(emotion)
    for continuous in pred_cont:
      write_line.append(str('%.4f' %(continuous)))
    with open(result_file, 'a') as f:
      f.writelines(' '.join(write_line))
      f.writelines('\n')

    selected_str = ', '.join([f'{e}:{p:.4f}' for e, p in selected_with_probs]) if selected_with_probs else '(none)'
    all_probs_str = ', '.join([f'{k}={v:.4f}' for k, v in probs_dict.items()])
    vad_str = ', '.join([f'{v:.4f}' for v in pred_cont])
    line_probs = f"{image_context_path} | SELECTED: {selected_str} | ALL: {all_probs_str} | VAD: {vad_str}"
    with open(result_file_probs, 'a') as f:
      f.write(line_probs + '\n')
