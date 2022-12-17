import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from models import GNN, AutoEncoder, SchNet, VariationalAutoEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from util import dual_CL
from pretrain_JOAO import graphcl_brics, graphcl3d_brics
from datasets import Molecule3DDatasetFragRandomaug3d_4frag
from models_mae import MaskedAutoencoderViT

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(gnn.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'mae': mae.state_dict()
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(gnn.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'mae': mae.state_dict()
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, mae):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    mae.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    loss_accum = 0.0
    ans1_accum=0
    ans2_accum=0
    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    
    
    for step, (batch, orig_batch, frag0, frag1, frag2, frag3) in enumerate(l):
        batch = batch.to(device)
        orig_batch = orig_batch.to(device)
        frag0 = frag0.to(device)
        frag1 = frag1.to(device)
        frag2 = frag2.to(device)
        frag3 = frag3.to(device)


        rep_2D_0 = molecule_model_2D.forward_mae(frag0.x, frag0.edge_index, frag0.edge_attr, frag0.batch).unsqueeze(1)
        rep_2D_1 = molecule_model_2D.forward_mae(frag1.x, frag1.edge_index, frag1.edge_attr, frag1.batch).unsqueeze(1)
        rep_2D_2 = molecule_model_2D.forward_mae(frag2.x, frag2.edge_index, frag2.edge_attr, frag2.batch).unsqueeze(1)
        rep_2D_3 = molecule_model_2D.forward_mae(frag3.x, frag3.edge_index, frag3.edge_attr, frag3.batch).unsqueeze(1)
        
        rep_3D_0 = molecule_model_3D.forward_mae(frag0.x[:, 0], frag0.positions , frag0.batch).unsqueeze(1)
        rep_3D_1 = molecule_model_3D.forward_mae(frag1.x[:, 0], frag1.positions , frag1.batch).unsqueeze(1)
        rep_3D_2 = molecule_model_3D.forward_mae(frag2.x[:, 0], frag2.positions , frag2.batch).unsqueeze(1)
        rep_3D_3 = molecule_model_3D.forward_mae(frag3.x[:, 0], frag3.positions , frag3.batch).unsqueeze(1)
        #molecule_2D_repr = molecule_model_2D.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        #molecule_2D_repr_mixed = molecule_model_2D.forward_frag_cl(frag0, frag1, frag2, frag3)       
        #loss_2D = molecule_model_2D.loss_cl(molecule_2D_repr, molecule_2D_repr_mixed)
        
        rep = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3, rep_3D_0, rep_3D_1, rep_3D_2, rep_3D_3], dim=1)
        
        loss, pred, mask, ans1, ans2 = mae(rep.detach())
        
        ans1_accum += ans1.detach().item()

        ans2_accum += ans2.detach().item()
        loss_accum += loss.detach().item()
        #if args.model_3d == 'schnet':
        #    molecule_3D_repr = molecule_model_3D.forward_cl(orig_batch.x[:, 0], orig_batch.positions , orig_batch.batch) #+ torch.randn_like(orig_batch.positions).cuda()
        #    molecule_3D_repr_mixed = molecule_model_3D.forward_frag_cl(frag0, frag1, frag2, frag3)       #+ torch.randn_like(orig_batch2.positions).cuda()
        #    loss_3D = molecule_model_3D.loss_cl(molecule_3D_repr, molecule_3D_repr_mixed)            
        

        #CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        
        
        #CL_loss_accum +=loss_2D.detach().cpu().item() + loss_3D.detach().cpu().item() + CL_loss.detach().cpu().item()
        #CL_acc_accum += CL_acc
        
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(molecule_model_2D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(molecule_model_3D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 5)
        optimizer.step()

    global optimal_loss

    temp_loss = loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('Loss: {:.5f}\t Acc1: {:.5f}\t Acc2: {:.5f}\t Time: {:.5f}'.format(
        loss_accum / float(len(l)), ans1_accum / float(len(l)), ans2_accum / float(len(l)), time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/molecule_datasets/GEOM_2D_3D_nmol{}_4frag_cut_singlebond_app'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug3d_4frag(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    molecule_model_2D = graphcl_brics(gnn).to(device)
    #molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout)
        molecule_model_3D = graphcl3d_brics(molecule_model_3D).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))


    molecule_model_2D.load_state_dict(torch.load(args.saved_complete_file)['model'])
    molecule_model_3D.load_state_dict(torch.load(args.saved_complete_file)['model_3D'])
    #for param in molecule_model_2D.parameters():
    #    param.requires_grad_(False)
    #for param in molecule_model_3D.parameters():
    #    param.requires_grad_(False)

    mae = MaskedAutoencoderViT(patch_size=16, in_chans=1,
                 embed_dim=300, depth=16, num_heads=10,
                 decoder_embed_dim=128, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)
    print(sum(p.numel() for p in molecule_model_2D.parameters()))
    
    print(sum(p.numel() for p in mae.parameters()))
#    if args.AE_model == 'AE':
#        AE_2D_3D_model = AutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
#        AE_3D_2D_model = AutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
#    elif args.AE_model == 'VAE':
#        AE_2D_3D_model = VariationalAutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
#        AE_3D_2D_model = VariationalAutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
#    else:
#        raise Exception

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * 0.01})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * 0.01})
    model_param_group.append({'params': mae.parameters(), 'lr': 0.1 * args.lr})

#    model_param_group.append({'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.gnn_lr_scale})
#    model_param_group.append({'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    #for epoch in range(1, args.epochs + 1):
    for epoch in range(1, 1000 + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, mae)

    save_model(save_best=False)
