from torch._C import device
from model import Generator_3 as Generator
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
import random
from curtsies.fmtfuncs import red, blue, green, yellow

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# imports for wav generation
import soundfile
from assets.autovc.synthesis import build_model
from assets.autovc.synthesis import wavegen

# use demo data for simplicity
# make your own validation set as needed
# validation set structure (from github question): [Speaker_Name , One-hot , [Mel, normed-F0, length, utterance_name]]
# also see here: https://github.com/auspicious3000/SpeechSplit/issues/11
# validation_pt = pickle.load(open('/Users/tobi/Data/corpora/CommonSpeakerSplit/de/val.pkl', 'rb'))
validation_pt = pickle.load(open('/DATA/tobi/CommonSpeakerSplit/en/val.pkl', 'rb'))

class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.audio_step = config.audio_step
        self.model_save_step = config.model_save_step
        

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

            
    def build_model(self):        
        self.G = Generator(self.hparams)
        
        self.Interp = InterpLnr(self.hparams)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        
        self.G.to(self.device)
        self.Interp.to(self.device)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)
        
        
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
#=====================================================================================================================
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
        
        # Print logs in specified order
        keys = ['G/loss_id']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
                # TODO: idea -> try and plot the spectrograms here (i.e. for each batch) so that I can better understand how it is structured
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            
            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)
            
                    
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)  # combines spect and f0s
            x_f0_intrp = self.Interp(x_f0, len_org) # random resampling with linear interpolation
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0] # strips f0 from trimmed to quantize it
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)
            
            x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean') 
           
            # Backward and optimize.
            g_loss = g_loss_id
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)
                        
                        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save({'model': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)
                print(green('Saved model checkpoints into {}...').format(self.model_save_dir))            
            

            # Validation.
            if (i+1) % self.sample_step == 0:
                self.G = self.G.eval()
                with torch.no_grad():
                    loss_val = []
                    # select X randomly chosen validation files
                    print(type(validation_pt))
                    validation_set = random.choice(list(validation_pt), 100)
                    for val_sub in validation_set:
                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)         
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 576)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device) 
                            f0_org = np.pad(val_sub[k][1], (0, 576-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            g_loss_val = F.mse_loss(x_real_pad, x_identic_val, reduction='sum')
                            loss_val.append(g_loss_val.item())
                val_loss = np.mean(loss_val) 
                print(red('Validation loss: {}').format(val_loss))
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_loss', val_loss, i+1)


            # plot val samples
            if (i+1) % self.sample_step == 0:
                print('Generating plot files ...')
                self.G = self.G.eval()
                with torch.no_grad():
                    maxPlotCount = 3
                    for val_sub in validation_set[0:maxPlotCount]:
                        speaker = val_sub[0]
                        speaker_fig_sample_dir = os.path.join(self.sample_dir, speaker, 'fig')
                        if not os.path.exists(speaker_fig_sample_dir):
                            os.makedirs(speaker_fig_sample_dir)

                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)         
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 576)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device) 
                            f0_org = np.pad(val_sub[k][1], (0, 576-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)
                            
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            x_identic_woF = self.G(x_f0_F, x_real_pad, emb_org_val)
                            x_identic_woR = self.G(x_f0, torch.zeros_like(x_real_pad), emb_org_val)
                            x_identic_woC = self.G(x_f0_C, x_real_pad, emb_org_val)
                            x_identic_woT = self.G(x_f0, x_real_pad, torch.zeros_like(emb_org_val))

                            melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                            melsp_gd_pad = melsp_gd_pad[:len_org]
                            melsp_out = x_identic_val[0].cpu().numpy().T
                            melsp_woF = x_identic_woF[0].cpu().numpy().T
                            melsp_woR = x_identic_woR[0].cpu().numpy().T
                            melsp_woC = x_identic_woC[0].cpu().numpy().T
                            melsp_woT = x_identic_woT[0].cpu().numpy().T
                            
                            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            
                            fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, 1, sharex=True, constrained_layout=True)
                            im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                            ax1.set_title('source')
                            im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                            ax2.set_title('output complete')
                            im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                            ax3.set_title('output without Content')
                            im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                            ax4.set_title('output without Rythm')
                            im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                            ax5.set_title('output without Pitch')
                            im6 = ax6.imshow(melsp_woT, aspect='auto', vmin=min_value, vmax=max_value)
                            ax6.set_title('output without Timbre')
                            plt.savefig(f'{speaker_fig_sample_dir}/{i+1}_{val_sub[0]}_{k}.png', dpi=600)
                            plt.close(fig) 


            # save mspecs, to be converted to a wav
            if (i+1) % self.audio_step == 0:
                print('Generating .wav files ...')
                self.G = self.G.eval()
                with torch.no_grad():
                    maxAudioCount = 4
                    for val_sub in validation_pt[0:maxAudioCount]:
                        speaker = val_sub[0]
                        speaker_audio_sample_dir = os.path.join(self.sample_dir, speaker, 'audio')
                        if not os.path.exists(speaker_audio_sample_dir):
                            os.makedirs(speaker_audio_sample_dir)

                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)         
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 576)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device) 
                            f0_org = np.pad(val_sub[k][1], (0, 576-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)
                            
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            x_identic_woF = self.G(x_f0_F, x_real_pad, emb_org_val)
                            x_identic_woR = self.G(x_f0, torch.zeros_like(x_real_pad), emb_org_val)
                            x_identic_woC = self.G(x_f0_C, x_real_pad, emb_org_val)
                            x_identic_woT = self.G(x_f0, x_real_pad, torch.zeros_like(emb_org_val))

                            # melsp_gd_pad = x_real_pad[0].cpu().numpy()
                            melsp_out = x_identic_val[0].cpu().numpy()    # this is the one to convert to audio 
                            # melsp_woF = x_identic_woF[0].cpu().numpy()
                            # melsp_woR = x_identic_woR[0].cpu().numpy()
                            # melsp_woC = x_identic_woC[0].cpu().numpy()
                            # melsp_woT = x_identic_woT[0].cpu().numpy()

                            # load pretraind vocoder from AutoVC
                            device = torch.device('cuda:0')
                            model = build_model().to(device)
                            checkpoint = torch.load('assets/checkpoint_step001000000_ema.pth', map_location=torch.device(torch.device(device)))
                            model.load_state_dict(checkpoint['state_dict'])

                            # convert output spect to wav (including padding)
                            waveform = wavegen(model, c=melsp_out)
                            soundfile.write(f'{speaker_audio_sample_dir}/{i+1}_{val_sub[0]}_{k}.wav', waveform, samplerate=16000)

                            

                            

