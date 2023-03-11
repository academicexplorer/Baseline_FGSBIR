from torch.autograd import Variable
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pytorch_lightning as pl


class FGSBIR_Model(pl.LightningModule):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(hp.backbone_name + '_Network(hp)')
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.hp = hp

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):

        positive_feature = self.sample_embedding_network(batch['positive_img'])
        negative_feature = self.sample_embedding_network(batch['negative_img'])
        sample_feature = self.sample_embedding_network(batch['sketch_img'])

        loss = self.loss(sample_feature, positive_feature, negative_feature)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sketch_feat, positive_feat = self.test_forward(batch)
        return sketch_feat, positive_feat, batch['sketch_path'], batch['positive_path']
    
    def validation_epoch_end(self, validation_step_outputs):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        
        for sketch_feat, positive_feat, sketch_path, positive_path in validation_step_outputs:
            Sketch_Feature_ALL.extend(sketch_feat)
            Sketch_Name.extend(sketch_path)

            for i_num, positive_name in enumerate(positive_path):
                if positive_name not in Image_Name:
                    Image_Name.append(positive_path[i_num])
                    Image_Feature_ALL.append(positive_feat[i_num])
        
        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        self.log('top1', top1)
        self.log('top10', top10)
        print ('Evaluation metrics: Top1 %.4f Top10 %.4f'%(top1, top10))

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sample_embedding_network(batch['sketch_img'])
        positive_feature = self.sample_embedding_network(batch['positive_img'])
        return sketch_feature, positive_feature



