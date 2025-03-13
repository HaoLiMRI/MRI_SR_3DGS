import os

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

from opt.option import args
# from data.LQGT_dataset import LQGTDataset
# from util.utils import RandCrop, RandHorizontalFlip, RandRotate, ToTensor, VGG19PerceptualLoss
from util.utils import init_random_seed, calc_psnr_for_mri_image
from util.reconstruction import high_res_image_rendering
from model.model_3d import GaussianSplattingNet
import pytorch_ssim_l1_org
from tqdm import tqdm
from datasets import get_eval_dataloader_3d, get_baseline_dataloader_3d
import scipy.io


# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()


if args.manual_seed is not None:
    init_random_seed(args.manual_seed)
    
args_file = open(os.path.join(args.snap_path, 'args.txt'), 'w')
for k, v in vars(args).items():
    args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')
args_file.close()

train_loader, validation_loader = get_baseline_dataloader_3d(args.dir_target_data, args.dir_target_data, args.batch_size)
len_data_loader = len(train_loader)

model = GaussianSplattingNet().cuda()

# loss
loss_L1 = nn.L1Loss().cuda()
# loss_pixel = nn.L1Loss().cuda()
# loss_MSE = nn.MSELoss().cuda()
# loss_adversarial = nn.BCEWithLogitsLoss().cuda()
# loss_percept = VGG19PerceptualLoss().cuda()
loss_ssim = pytorch_ssim_l1_org.SSIM().cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr_G,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
    )
    
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-8, last_epoch = -1)

if args.checkpoint:
    checkpoint = torch.load(args.weights)

    model(checkpoint['model'])

#    optimizer.load_state_dict(checkpoint['optimizer'])

#    scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0

log_file = open(os.path.join(args.snap_path, 'log.txt'), 'w')
print('Training starts:')
max_ssim = 0.0
# training
for epoch in range(start_epoch, args.epochs):
    
    model.train()
    running_loss_l1 = 0.0
    running_loss_ssim = 0.0
    total_loss = 0.0
    
    for step, (LR, HR) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        
        LR, HR = LR.cuda(), HR.cuda()

        outputs = model(LR)
        SR_img = high_res_image_rendering(outputs)
        l1_loss = loss_L1(SR_img.unsqueeze(1), HR)
        ssim_loss = 1 - loss_ssim(SR_img, HR.squeeze(1)).pow(2)
        loss = l1_loss + ssim_loss
#        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss_l1 += l1_loss.item()
        running_loss_ssim += ssim_loss.item()
            
        if ((step + 1) % args.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]:"
                      "loss_D_total={:.5f} loss_L1={:.5f} loss_SSIM={:.5f}"
                      .format(epoch + 1,
                              args.epochs,
                              step + 1,
                              len_data_loader,
                              optimizer.param_groups[0]['lr'],
                              total_loss/args.log_step,
                              running_loss_l1/args.log_step,
                              running_loss_ssim/args.log_step))
            running_loss_l1 = 0.0
            running_loss_ssim = 0.0
            total_loss = 0.0
    
    scheduler.step()
        
    # Validation Step
    model.eval()
    total_psnr, total_ssim = 0, 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs.cuda())
            SR_img = high_res_image_rendering(outputs)
            total_ssim += loss_ssim(SR_img, targets.squeeze(1).cuda()).item()
            total_psnr += calc_psnr_for_mri_image(SR_img.unsqueeze(1), targets.cuda()).item()

        
        avg_psnr = total_psnr / len(validation_loader)
        avg_ssim = total_ssim / len(validation_loader)
        print("Avg SSIM = {}, Avg PSNR = {}".format(avg_ssim, avg_psnr))
        log_file.write("Epoch: %d, SSIM: %f, PSNR: %f \n" % (epoch+1, avg_ssim, avg_psnr))
        
        
    if epoch == 0 or max_ssim<avg_ssim:
        max_ssim = avg_ssim
        weights_file_name = 'best_ssim_network_parameter.pth'
        weights_file = os.path.join(args.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,

            'model': model.state_dict(),

#            'optimizer': optimizer_D.state_dict(),

#            'scheduler': scheduler_D.state_dict(),
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))
        log_file.write("Network saved! \n")
    log_file.write("\n")
log_file.close()        

if args.perform_inference:
    "Evaluation(Test)"
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    filenames = sorted(os.listdir(os.path.join(args.dir_test,'Evaluation')))
    for filename in tqdm(filenames):
        if 'data1.mat' in filename:
            eval_loader = get_eval_dataloader_3d(args.dir_test, filename, args.batch_size)
            with torch.no_grad():    
                for i, tgt_LR in enumerate(eval_loader, 0):
                    LR_img = tgt_LR[0].float().cuda()
                    outputs = model(LR_img)
                    SR_img = high_res_image_rendering(outputs)
                    if i==0:
                        SR_img_eval_tensor = SR_img.data
                    else:
                        SR_img_eval_tensor = torch.cat((SR_img_eval_tensor, SR_img.data), 0)
    
            SR_images_test = SR_img_eval_tensor.cpu().numpy()
            "save the .mat files for SR, HR and LR training images"
            scipy.io.savemat(os.path.join(args.results, os.path.splitext(filename)[0]+'_SR_test_image.mat'), mdict = {'SR_test_image' : SR_images_test})
