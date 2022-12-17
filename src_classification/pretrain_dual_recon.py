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
from torch_geometric.utils import to_dense_adj

from datasets import Molecule3DMaskingDataset


class Dist_Projector(nn.Module):
    def __init__(self):
        super(Dist_Projector, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(600, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )
    def forward(self, x):
        return self.projection_head(x)

class Edge_Projector(nn.Module):
    def __init__(self):
        super(Edge_Projector, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(600, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )
    def forward(self, x):
        return self.projection_head(x)

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    edge_acc_accum = 0
    total_edge_accum = 0
    dist_loss_accum = 0.0
    edge_loss_accum = 0.0
    dist_criterion = nn.MSELoss()
    edge_criterion = nn.BCEWithLogitsLoss()
    

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_repr_2D = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr_2D, batch.batch)

        if args.model_3d == 'schnet':
            node_repr_3D, molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch, node_feat=True)


        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        molecule_size = torch.bincount(batch.batch)
        mol_idx = 0
            
        #pred_dist_mat_input = torch.zeros(args.batch_size, torch.max(molecule_sizes), torch.max(molecule_sizes), args.emb_dim*2, device=device)
        #pred_dist_mat_mask = torch.zeros(args.batch_size, torch.max(molecule_sizes), torch.max(molecule_sizes), device=device)
        #true_dist_mat = torch.zeros(args.batch_size, torch.max(molecule_sizes), torch.max(molecule_sizes), device=device)
        
        batch_edge_mat = to_dense_adj(batch.edge_index, batch.batch)

        dist_loss = 0.0
        edge_loss = 0.0
        for b in range(len(molecule_size)):
            
            position = batch.positions[mol_idx : mol_idx + molecule_size[b],:]

            pos_1 = position.unsqueeze(1)
            pos_2 = position.unsqueeze(0)

            a2 = (pos_1 ** 2).sum(dim=2, keepdims=True)
            b2 = a2.transpose(0,1)

            ab = pos_2.matmul(pos_2.transpose(-1,-2)).permute(1,2,0)
            true_dist_mat = torch.sqrt(a2 + b2 - 2*ab).squeeze()
            
            features_2D = node_repr_2D[mol_idx : mol_idx + molecule_size[b],:]
            
            features_2D_1 = features_2D.repeat(molecule_size[b],1).reshape(molecule_size[b], molecule_size[b], -1)
            features_2D_2 = features_2D.repeat_interleave(molecule_size[b], dim=0).reshape(molecule_size[b], molecule_size[b], -1)

            features_2D = torch.cat((features_2D_1, features_2D_2),-1)
            
            pred_dist_mat = dist_projector(features_2D).squeeze()
            pred_dist_mat = pred_dist_mat + pred_dist_mat.transpose(0,1)
            pred_dist_mat = torch.log(1.0 + torch.exp(pred_dist_mat))
            #final_pred_dist_mat = torch.triu(pred_dist_mat, diagonal=1)
            #final_dist_mat = torch.triu(true_dist_mat, diagonal=1)
            final_pred_dist_mat = pred_dist_mat.flatten()[1:].view(molecule_size[b]-1, molecule_size[b]+1)[:,:-1].reshape(molecule_size[b], molecule_size[b]-1)
            final_dist_mat = true_dist_mat.flatten()[1:].view(molecule_size[b]-1, molecule_size[b]+1)[:,:-1].reshape(molecule_size[b], molecule_size[b]-1)


            
            true_edge_mat = batch_edge_mat[b,:molecule_size[b],:molecule_size[b]]

            features_3D = node_repr_3D[mol_idx : mol_idx + molecule_size[b],:]
            
            features_3D_1 = features_3D.repeat(molecule_size[b],1).reshape(molecule_size[b], molecule_size[b], -1)
            features_3D_2 = features_3D.repeat_interleave(molecule_size[b], dim=0).reshape(molecule_size[b], molecule_size[b], -1)

            features_3D = torch.cat((features_3D_1, features_3D_2),-1)

            pred_edge_mat = edge_projector(features_3D).squeeze()
            final_pred_edge_mat = (pred_edge_mat + pred_edge_mat.transpose(0,1)).flatten()[1:].view(molecule_size[b]-1, molecule_size[b]+1)[:,:-1].reshape(molecule_size[b], molecule_size[b]-1)
            
            final_edge_mat = true_edge_mat.flatten()[1:].view(molecule_size[b]-1, molecule_size[b]+1)[:,:-1].reshape(molecule_size[b], molecule_size[b]-1)

            mol_idx += molecule_size[b]

            # true_dist_mat[b, offset:offset+molecule_size[b], offset:offset+molecule_size[b]] =     # max(molecule_size) x max(molecule_size) x (emb_dim*2)
            
            # poisition = batch.positions[mol_idx:offset_] # molecule_sizes[b] x dim

            # pos_1 = poisition.unsqueeze(1) # molecule_sizes[b] x 1 x dim
            # pos_2 = poisition.unsqueeze(0) # 1 x molecule_sizes[b] x dim

            # a2 = (pos_1 ** 2).sum(dim=2, keepdims=True) # molecule_sizes[b] x 1 x 1
            # b2 = a2.transpose(0, 1) # 1 x molecule_sizes[b] x 1

            # # b2 = (pos_2 ** 2).sum(dim=2, keepdims=True) # 1 x molecule_sizes[b] x 1

            # ab = pos_2.matmul(pos_2.transpose(-1, -2)).permute(1, 2, 0) # molecule_sizes[b] x molecule_sizes[b] x 1               
            # result = torch.sqrt(a2 + b2 - 2*ab) # molecule_sizes[b] x molecule_sizes[b] x 1     

            # true_dist_mat[b, offset:offset+molecule_size[b], offset:offset+molecule_size[b]] = result




            # (a-b)**2 = a**2 + b**2 - 2*a*b
            
            # torch.cdist(molecule_2D_repr[offset_+molecule_sizes[b], :] # molecule_sizes[b] x dim

            #torch.norm(batch.positions[mol_idx + i]-batch.positions[mol_idx+j])


            
            #pred_dist_mat_input[b]
            
            #for i in range(molecule_sizes[b]):
            #    for j in range(molecule]_sizes[b]):
            #        true_dist_mat[b,i,j] = torch.norm(batch.positions[mol_idx + i]-batch.positions[mol_idx+j])
            #        pred_dist_mat_input[b,i,j,:args.emb_dim] = molecule_2D_repr[mol_idx + i, :]
            #        pred_dist_mat_input[b,i,j,args.emb_dim:] = molecule_2D_repr[mol_idx + j, :]
            #pred_dist_mat_mask[b, :molecule_sizes[b], :molecule_sizes[b]] = torch.ones(molecule_sizes[b], molecule_sizes[b]).to(device)
            


            dist_loss += dist_criterion(final_pred_dist_mat, final_dist_mat)
            edge_loss += edge_criterion(final_pred_edge_mat, final_edge_mat)

            with torch.no_grad():
                edge_pred_tag = torch.round(torch.sigmoid(final_pred_edge_mat))
                #print(y_pred_tag, is_permuted)
                edge_acc_accum += (edge_pred_tag == final_edge_mat.float()).float().sum()
                total_edge_accum += torch.numel(final_edge_mat)
                #acc = correct_results_sum/is_permuted.unsqueeze(1).float().shape[0]
                #print(dist_loss)
        #loss_2d_to_3d = avg_dist_loss(node_repr, batch.)
        
        
        #AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
        #AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
        #AE_acc_1 = AE_acc_2 = 0
        #AE_loss = (AE_loss_1 + AE_loss_2) / 2

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        #AE_loss_accum += AE_loss.detach().cpu().item()
        #AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2
        dist_loss = dist_loss / len(batch.batch)
        dist_loss_accum += dist_loss.detach().cpu().item()

        edge_loss = edge_loss / len(batch.edge_attr)
        edge_loss_accum += edge_loss.detach().cpu().item()
        #if args.alpha_1 > 0:
        loss = CL_loss + dist_loss + edge_loss
        #if args.alpha_2 > 0:
        #    loss += AE_loss * args.alpha_2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(molecule_model_2D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(molecule_model_3D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(edge_projector.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(dist_projector.parameters(), 5)
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    edge_acc_accum /= total_edge_accum
    dist_loss_accum /= len(loader)
    temp_loss = CL_loss_accum + dist_loss_accum + edge_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tdist Loss: {:.5f}\tedge loss: {:.5f}\tedge Acc: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, dist_loss_accum, edge_loss_accum, edge_acc_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        data_root = '../datasets/{}/'.format(args.dataset) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset)
        dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio)
    else:
        raise Exception
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))
    dist_projector = Dist_Projector().to(device)
    edge_projector = Edge_Projector().to(device)
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
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    model_param_group.append({'params': dist_projector.parameters(), 'lr': args.lr})
    model_param_group.append({'params': edge_projector.parameters(), 'lr': args.lr})
#    model_param_group.append({'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.gnn_lr_scale})
#    model_param_group.append({'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, device, loader, optimizer)

    save_model(save_best=False)
