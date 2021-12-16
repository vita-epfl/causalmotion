import torch
import torch.nn as nn
from tqdm import tqdm

#from models import SimpleDecoder, SimpleEncoder, SimpleStyleEncoder
from utils import NUMBER_PERSONS
from losses import standard_style_loss

class ConcatBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConcatBlock, self).__init__()
        self.perceptron = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

    def forward(self, x, style):
        if style == None:
            return x
            
        B, D = x.shape
        content_and_style = torch.cat((x, style.repeat(B, 1)), dim=1)
        out = self.perceptron(content_and_style)
        return out + x


class SimpleStyleEncoder(nn.Module):
    def __init__(self, args):
        super(SimpleStyleEncoder, self).__init__()

        # style encoder
        self.encoder = nn.Sequential(
            nn.Linear(40, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.args = args
        self.style_dim = 8

        # style classifier
        hidden_size = 8 #50
        feat_space_dim = 8

        # hat classifier above style to get a low dim space for contrastive learning
        self.hat_classifier = nn.Sequential(
            nn.Linear(self.style_dim, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, feat_space_dim)
        )

        # classifier to get error_rate of learned latent space
        self.er_classifier = nn.Sequential(
            nn.Linear(feat_space_dim, feat_space_dim),
            nn.ReLU(), nn.Linear(feat_space_dim, args.classification)
        )
        # associated optimizer. Trained by used train_er_classifier()
        self.er_opt = torch.optim.Adam(self.er_classifier.parameters())


    def train_er_classifier(self,  train_dataset):
        assert (self.args.contrastive)

        for e in tqdm(range(3)):
            train_loaders_iter = [iter(train_loader) for train_loader in train_dataset['loaders']]
            for _ in range(train_dataset['num_batches']):
                batch_loss = []
                self.er_opt.zero_grad() # reset gradients

                for train_loader_iter, loader_name in zip(train_loaders_iter, train_dataset['names']):
                    batch = next(train_loader_iter)
                    with torch.no_grad():
                        low_dim = self.forward(batch[5].cuda(), 'low')
                    class_preds = self.er_classifier(low_dim)
                    label = train_dataset['labels'][loader_name]
                    batch_loss.append(standard_style_loss(class_preds, label))
                
                loss = torch.stack(batch_loss).sum()
                loss.backward()
                self.er_opt.step()


    def forward(self, style_input, what):
        assert(what in set(['low', 'both', 'style', 'class']))
        # for batch size 68
        # style 20 x 128 x 2
        style_input = torch.stack(style_input.split(2, dim=1), dim=1)[:,:,1,:] # 20 x 64 x 2
        style_input = torch.permute(style_input, (1, 0, 2))  # 64 x 20 x 2
        style_input = torch.flatten(style_input, 1) # 64 x 40

        # MLP
        style_seq = self.encoder(style_input)

        # apply reduction without sequences / within batch if needed
        batch_style = style_seq.mean(dim=0).unsqueeze(dim=0)

        if 'style' == what: # only what the style
            return batch_style

        low_seq = self.hat_classifier(style_seq)
        if low_seq.dim()==1: low_seq = torch.nn.functional.normalize(low_seq, dim=0)
        else: low_seq = torch.nn.functional.normalize(low_seq)

        if 'low' == what: # only what the contrastive feat space
            return low_seq
        elif 'both' == what: # both what
            return low_seq, batch_style
        elif 'class' == what:
            class_out = self.er_classifier(low_seq) # only what the class label
            return class_out
        else:
            raise NotImplementedError


class SimpleEncoder(nn.Module):
    def __init__(
            self,
            obs_len,
            hidden_size,
            number_agents
    ):
        super(SimpleEncoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len*number_agents*2, hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size*2),
        )


    def forward(self, obs_traj_rel):

        obs_traj_rel = torch.stack(obs_traj_rel.split(2, dim=1), dim=1)
        obs_traj_rel = torch.permute(obs_traj_rel, (1, 2, 0, 3))
        obs_traj_rel = obs_traj_rel.flatten(start_dim=1)

        encoded = self.mlp(obs_traj_rel)
        
        encoded = torch.stack(encoded.split(encoded.shape[1]//2, dim=1), dim=1)
        encoded = encoded.flatten(start_dim=0, end_dim=1)
        
        return encoded





class SimpleDecoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            hidden_size,
            number_of_agents,
            style_input_size=None,
    ):
        super(SimpleDecoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len
        self.fut_len = fut_len
        
        self.style_input_size = style_input_size

        self.noise_fixed = False

        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size*2, 4* hidden_size),
            nn.ReLU(),
            nn.Linear(4* hidden_size, 4* hidden_size)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(4* hidden_size, number_of_agents*2*fut_len),
            nn.ReLU(),
            nn.Linear(number_of_agents*2*fut_len, number_of_agents*2*fut_len)
        )

        self.number_of_agents = number_of_agents

        self.style_blocks = nn.ModuleList(
            [ConcatBlock(self.style_input_size + hidden_size*2, hidden_size*2),
            ConcatBlock(self.style_input_size + 4* hidden_size, 4* hidden_size)]
        )


    def forward(self, latent_space, style_feat_space=None):

        traj_lstm_hidden_state = torch.stack(latent_space.split(2, dim=0), dim=0)
        out = traj_lstm_hidden_state.flatten(start_dim=1)

        if style_feat_space != None:
            out = self.style_blocks[0](out, style_feat_space)

        out = self.mlp1(out)

        if style_feat_space != None:
            out = self.style_blocks[1](out, style_feat_space)

        out = self.mlp2(out)

        out = torch.reshape(out, (out.shape[0], self.number_of_agents, self.fut_len, 2))

        out = out.flatten(start_dim=0, end_dim=1)

        out = torch.permute(out, (1, 0, 2))

        return out


class CausalMotionModel(nn.Module):
    def __init__(self, args):
        super(CausalMotionModel, self).__init__()

        latent_space_size = 8

        self.inv_encoder = SimpleEncoder(args.obs_len, latent_space_size, NUMBER_PERSONS)
        self.style_encoder = SimpleStyleEncoder(args)

        self.decoder = SimpleDecoder(
            args.obs_len,
            args.fut_len,
            latent_space_size,
            NUMBER_PERSONS,
            style_input_size=self.style_encoder.style_dim,
        )

    def forward(self, batch, training_step):
        assert (training_step in ['P3', 'P4', 'P5', 'P6'])

        (obj_traj, _, _, _, _, style_input, _) = batch

        # compute only style and classify
        if training_step == 'P4':
            if self.training:
                return self.style_encoder(style_input, 'low')
            else:
                return self.style_encoder(style_input, 'class')

        # compute invariants
        latent_content_space = self.inv_encoder(obj_traj)

        # compute style if required
        style_encoding = None
        if training_step in ['P5', 'P6']:
            low_dim, style_encoding = self.style_encoder(style_input, 'both')

        # compute prediction
        output = self.decoder(latent_content_space, style_feat_space=style_encoding)

        if training_step == 'P6' and self.training:
            return output, low_dim  # need the low_dim to keep training contrastive loss
        else:
            return output