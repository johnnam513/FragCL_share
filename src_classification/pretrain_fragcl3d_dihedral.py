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
from pretrain_JOAO import graphcl, graphcl3d
from datasets import Molecule3DDatasetFragRandomaug3d
#from dig.threedgraph.method import SphereNet


#class SphereNetFrag(SphereNet):
#    def __init__(self, energy_and_force, cutoff, num_layers, hidden_channels, out_channels, int_emb_size, basis_emb_size_dist,
#                    basis_emb_size_angle, basis_emb_size_torsion, out_emb_channels, num_spherical, num_radial, envelope_exponent,
#                    num_before_skip, num_after_skip, num_output_layers):
#        super(SphereNet, self).__init__(energy_and_force, cutoff, num_layers, hidden_channels, out_channels, int_emb_size, basis_emb_size_dist,
#                    basis_emb_size_angle, basis_emb_size_torsion, out_emb_channels, num_spherical, num_radial, envelope_exponent,
#                    num_before_skip, num_after_skip, num_output_layers)

#    def 

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(gnn.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(gnn.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
             
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, (batch, batch1, batch2, orig_batch, orig_batch1, orig_batch2) in enumerate(l):
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        orig_batch.to(device)
        orig_batch1.to(device)
        orig_batch2.to(device)
        #orig_batch.anchor1_ang = orig_batch.anchor1_ang.view(-1,3)
        #orig_batch.anchor2_ang = orig_batch.anchor2_ang.view(-1,3)
        #print(batch.anchor2_ang.shape)
        #print(batch.anchor1_angle.shape)
        #print(batch.anchor2_angle.shape)
        

        _, _, molecule_2D_repr = molecule_model_2D.forward_cl_node(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        node_rep1, _, _ = molecule_model_2D.forward_cl_node(orig_batch1.x, orig_batch1.edge_index, orig_batch1.edge_attr, orig_batch1.batch)
        node_rep2, _, _ = molecule_model_2D.forward_cl_node(orig_batch2.x, orig_batch2.edge_index, orig_batch2.edge_attr, orig_batch2.batch)
        molecule_2D_repr_mixed, frag_2D_repr1, frag_2D_repr2 = molecule_model_2D.weighted_forward_frag_cl(batch1.x, batch1.edge_index,
                                batch1.edge_attr, batch1.batch,
                                batch2.x, batch2.edge_index,
                                batch2.edge_attr, batch2.batch)       
        loss_2D = molecule_model_2D.weighted_loss_cl_onlyneg(molecule_2D_repr, molecule_2D_repr_mixed, frag_2D_repr1, frag_2D_repr2, batch1.batch, batch2.batch)

        if args.model_3d == 'schnet' or args.model_3d == 'spherenet':
            molecule_3D_repr = molecule_model_3D.forward_cl(orig_batch.x[:, 0], orig_batch.positions + (torch.randn_like(orig_batch.positions).cuda()), orig_batch.batch) #+ torch.randn_like(orig_batch.positions).cuda()
            molecule_3D_repr_mixed, frag_3D_repr1, frag_3D_repr2 = molecule_model_3D.weighted_forward_frag_cl(orig_batch1.x[:,0], orig_batch1.positions +  (torch.randn_like(orig_batch1.positions).cuda()), orig_batch1.batch, #+ torch.randn_like(orig_batch1.positions).cuda()
                                orig_batch2.x[:,0], orig_batch2.positions + (torch.randn_like(orig_batch2.positions).cuda()), orig_batch2.batch)       #+ torch.randn_like(orig_batch2.positions).cuda()
            loss_3D = molecule_model_3D.weighted_loss_cl_onlyneg(molecule_3D_repr, molecule_3D_repr_mixed, frag_3D_repr1, frag_3D_repr2, orig_batch1.batch, orig_batch2.batch)            
        

        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        
        binc = torch.bincount(orig_batch.batch)
        binc1 = torch.bincount(orig_batch1.batch)
        binc2 = torch.bincount(orig_batch2.batch)

        batch_size = len(binc)
        
        offset1 =  torch.cat([torch.Tensor([0]).long().cuda(), torch.cumsum(binc1, dim=0)[:-1]], dim=0).unsqueeze(1).repeat(1,2)
        offset2 =  torch.cat([torch.Tensor([0]).long().cuda(), torch.cumsum(binc2, dim=0)[:-1]], dim=0).unsqueeze(1).repeat(1,2)
        
        anchor1 = offset1 + orig_batch.dihedral_anchor[:,:2]
        anchor2 = offset2 + orig_batch.dihedral_anchor[:,2:]

        anchor_from_frag1 = torch.index_select(node_rep1, 0, anchor1.view(-1)).reshape(batch_size, 2, 300)
        anchor_from_frag2 = torch.index_select(node_rep2, 0, anchor2.view(-1)).reshape(batch_size, 2, 300)
        
        anchor = torch.cat([anchor_from_frag1, anchor_from_frag2], dim=1)
        
        anchor_reverse = torch.zeros_like(anchor).cuda()

        anchor_reverse[:,0,:] = anchor[:,3,:]
        anchor_reverse[:,1,:] = anchor[:,2,:]
        anchor_reverse[:,2,:] = anchor[:,1,:]
        anchor_reverse[:,3,:] = anchor[:,0,:]
        
        anchor = anchor.reshape(batch_size,-1)
        anchor_out = molecule_model_2D.forward_aux2(anchor)
        loss_anchor = molecule_model_2D.ce2(anchor_out, batch.dihedral_angle)

        anchor_reverse = anchor_reverse.reshape(batch_size, -1)
        anchor_out2 = molecule_model_2D.forward_aux2(anchor_reverse)
        dihedral_angle2 = torch.remainder(batch.dihedral_angle + 9, 18).long()
        loss_anchor_reverse = molecule_model_2D.ce2(anchor_out2, dihedral_angle2)
        acc = (anchor_out.argmax(dim=1) == batch.dihedral_angle).sum() / batch_size
        acc2 = (anchor_out2.argmax(dim=1) == dihedral_angle2).sum() / batch_size
        acc = (acc + acc2) /2.0

        loss_anchor = (loss_anchor + loss_anchor_reverse)
        
        CL_loss_accum += CL_loss.detach().cpu().item()#CL_loss.detach().cpu().item() #+ loss_2D.detach().cpu().item() + loss_3D.detach().cpu().item()
        CL_acc_accum += CL_acc
        AE_loss_accum += loss_anchor.detach().cpu().item()
        AE_acc_accum += acc

        loss = 0
        if epoch < 5:
            loss += loss_2D + loss_3D +  loss_anchor 
        else:
            loss += loss_2D + CL_loss +  loss_3D + loss_anchor 
        
        
        #if args.alpha_2 > 0:
        #    loss += AE_loss * args.alpha_2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(molecule_model_2D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(molecule_model_3D.parameters(), 5)
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tAE Acc: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, AE_loss_accum, AE_acc_accum, time.time() - start_time))
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
        root_2d = '{}/GEOM_2D_3D_nmol{}_singlebond_manual_1207_sorted_dihedral_from_frag_nohydanchor'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug3d(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    if 'QM9' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/QM9_2D_3D_cut_singlebond_manual'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug3d(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)
    

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    molecule_model_2D = graphcl(gnn).to(device)
    

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout)
        molecule_model_3D = graphcl3d(molecule_model_3D).to(device)
    elif args.model_3d == 'spherenet':
        #molecule_model_3D = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
        #          hidden_channels=128, out_channels=1, int_emb_size=64,
        #          basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        #          num_spherical=3, num_radial=6, envelope_exponent=5,
        #          num_before_skip=1, num_after_skip=2, num_output_layers=3)   
        molecule_model_3D = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=300, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)   
        molecule_model_3D = graphcl3d(molecule_model_3D).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10
    epoch = 0
    for epoch in range(1, args.epochs + 1):
    #while epoch < 100:
        print('epoch: {}'.format(epoch))
        
        train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch) ###
    #        epoch += 1
        
            

    save_model(save_best=False)
