#!/usr/bin/env python

"""Module with Corpus class. There are methods for each step of the alg for the
whole video collection of one complex activity. See pipeline."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np
import random
import os
from os.path import join
import os.path as ops
import torch
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture
import time
from sklearn.cluster import MiniBatchKMeans

from ute.video import Video
from ute.models import mlp
from ute.models import cls
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.eval_utils.accuracy_class import Accuracy
from ute.utils.mapping import GroundTruth
from ute.utils.util_functions import join_data, timing, dir_check
from ute.utils.visualization import Visual, plot_segm
from ute.probabilistic_utils.gmm_utils import AuxiliaryGMM, GMM_trh
from ute.eval_utils.f1_score import F1Score
from ute.models.dataset_loader import load_reltime, load_pseudo_gt, load_single_video
from ute.models.training_embed import load_model, training, training_cls
from ute.viterbi_utils.grammar import SingleTranscriptGrammar
from ute.viterbi_utils.length_model import PoissonModel
from ute.viterbi_utils.viterbi_w_lenth import Viterbi


class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, features, transcript, framelabels):
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # # update frame selectors
        # self.frame_selectors = []
        # for seq_idx in range(len(self.features)):
        #     self.frame_selectors += [ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ]

    # def random(self):
    #     return random.choice(self.frame_selectors) # return sequence_idx and frame_idx within the sequence

    # def n_frames(self):
    #     return len(self.frame_selectors)

class Corpus(object):
    def __init__(self, subaction='coffee', K=None, buffer_size=2000, frame_sampling=30, mean_lengths_file=None, prior_file=None):
        """
        Args:
            Q: number of Gaussian components in each mixture
            subaction: current name of complex activity
        """
        np.random.seed(opt.seed)
        self.gt_map = GroundTruth(frequency=opt.frame_frequency)
        self.gt_map.load_mapping()
        self._K = self.gt_map.define_K(subaction=subaction) if K is None else K
        logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.iter = 0
        self.return_stat = {}
        self._frame_sampling = frame_sampling

        self._acc_old = 0
        self._videos = []
        self._subaction = subaction
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        self._gaussians = {}
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features = None
        self._embedded_feat = None
        self._init_videos()
        # logger.debug('min: %f  max: %f  avg: %f' %
        #              (np.min(self._features),
        #               np.max(self._features),
        #               np.mean(self._features)))

        # to save segmentation of the videos
        dir_check(os.path.join(opt.output_dir, 'segmentation'))
        dir_check(os.path.join(opt.output_dir, 'likelihood'))
        self.vis = None  # visualization tool

        self.decoder = Viterbi(None, None, self._frame_sampling, max_hypotheses = np.inf)

        self.buffer = Buffer(buffer_size, self._K)
        if mean_lengths_file is None:            
            self.mean_lengths = np.ones((self._K), dtype=np.float32) * self._frame_sampling * 2
        else:
            self.mean_lengths = np.loadtxt(mean_lengths_file)
        
        if prior_file is None:
            self.prior = np.ones((self._K), dtype=np.float32) / self._K
        else:
            self.prior = np.loadtxt(prior_file)

    def _init_videos(self):
        logger.debug('.')
        gt_stat = Counter()
        for root, dirs, files in os.walk(opt.data):
            if not files:
                continue
            for filename in files:
                # pick only videos with certain complex action
                # (ex: just concerning coffee)
                if self._subaction in filename:
                    if opt.test_set:
                        if opt.reduced:
                            opt.reduced = opt.reduced - 1
                            continue
                    # if opt.dataset == 'fs':
                    #     gt_name = filename[:-(len(opt.ext) + 1)] + '.txt'
                    # else:
                    match = re.match(r'(.*)\..*', filename)
                    gt_name = match.group(1)
                    # use extracted features from pretrained on gt embedding
                    if opt.load_embed_feat:
                        path = os.path.join(opt.data, 'embed', opt.subaction,
                                            opt.resume_str % opt.subaction) + '_%s' % gt_name
                    else:
                        path = os.path.join(root, filename)
                    start = 0 if self._features is None else self._features.shape[0]
                    try:
                        video = Video(path, K=self._K,
                                      gt=self.gt_map.gt[gt_name],
                                      name=gt_name,
                                      start=start,
                                      with_bg=self._with_bg)
                    except AssertionError:
                        logger.debug('Assertion Error: %s' % gt_name)
                        continue
                    self._features = join_data(self._features, video.features(),
                                               np.vstack)

                    video.reset()  # to not store second time loaded features
                    self._videos.append(video)
                    # accumulate statistic for inverse counts vector for each video
                    gt_stat.update(self.gt_map.gt[gt_name])
                    if opt.reduced:
                        if len(self._videos) > opt.reduced:
                            break

                    if opt.feature_dim > 100:
                        if len(self._videos) % 20 == 0:
                            logger.debug('loaded %d videos' % len(self._videos))

        # update global range within the current collection for each video
        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: %d videos ' % len(self._videos) + str(gt_stat))
        self._update_fg_mask()

    def _update_fg_mask(self):
        logger.debug('.')
        if self._with_bg:
            self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
            for video in self._videos:
                self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
        else:
            self._total_fg_mask = np.ones(len(self._features), dtype=bool)

    def get_videos(self):
        for video in self._videos:
            yield video

    def get_features(self):
        return self._features

    def video_byidx(self, idx):
        return np.asarray(self._videos)[idx]

    def __len__(self):
        return len(self._videos)

    def regression_training(self):
        if opt.load_embed_feat:
            logger.debug('load precomputed features')
            self._embedded_feat = self._features
            return

        logger.debug('.')

        dataloader = load_reltime(videos=self._videos,
                                  features=self._features)

        model, loss, optimizer = mlp.create_model()
        if opt.load_model:
            model.load_state_dict(load_model())
            self._embedding = model
        else:
            self._embedding = training(dataloader, opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)

        self._embedding = self._embedding.cpu()

        unshuffled_dataloader = load_reltime(videos=self._videos,
                                             features=self._features,
                                             shuffle=False)

        gt_relative_time = None
        relative_time = None
        if opt.model_name == 'mlp':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                if self._embedded_feat is None:
                    self._embedded_feat = batch_features
                else:
                    self._embedded_feat = torch.cat((self._embedded_feat, batch_features), 0)

                batch_gtreltime = batch_gtreltime.numpy().reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)

            relative_time = self._embedding(self._embedded_feat.float()).detach().numpy().reshape((-1, 1))

            self._embedded_feat = self._embedding.embedded(self._embedded_feat.float()).detach().numpy()
            self._embedded_feat = np.squeeze(self._embedded_feat)

        if opt.save_embed_feat:
            self.save_embed_feat()

        mse = np.sum((gt_relative_time - relative_time)**2)
        mse = mse / len(relative_time)
        logger.debug('MLP training: MSE: %f' % mse)


    def train_classifier(self, video=None):
        logger.debug('train framewise classifier')
        # train_classifier
        logger.debug('.')

        if video == None:
            dataloader = load_pseudo_gt(videos=self._videos,
                                    features=self._embedded_feat,
                                    pseudo_gt=self.pseudo_gt_with_bg)
            num_epoch = 15
        else:
            dataloader = load_single_video(videos=self._videos,
                                    features=self._embedded_feat,
                                    pseudo_gt=self.pseudo_gt_with_bg,
                                    video=video)
            num_epoch = 5

        model, loss, optimizer = cls.create_model(self._K)

        self._classifier = training_cls(dataloader, num_epoch,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)
        # update video likelihood
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)
        

    def _video_likelihood_grid(self, video_idx):
        video = self._videos[video_idx]
        if opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
       
        scores = self._classifier(torch.FloatTensor(features).cuda()).cpu().detach().numpy()
        video._likelihood_grid = scores
        if opt.save_likelihood:
            video.save_likelihood()

 

    def clustering(self):
        logger.debug('.')
        np.random.seed(opt.seed)

        kmean = MiniBatchKMeans(n_clusters=self._K,
                                 random_state=opt.seed,
                                 batch_size=50)

        kmean.fit(self._embedded_feat[self._total_fg_mask])

        accuracy = Accuracy()
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
        long_rt = np.array(long_rt)

        kmeans_labels = np.asarray(kmean.labels_).copy()
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        logger.debug('time ordering of labels')
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            kmeans_labels[kmean.labels_ == label] = time_idx

        shuffle_labels = np.arange(len(time2label))

        self.pseudo_gt_with_bg = np.ones(len(self._total_fg_mask)) * -1

        # use predefined by time order  for kmeans clustering
        self.pseudo_gt_with_bg[self._total_fg_mask] = kmeans_labels

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = self.pseudo_gt_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        if opt.vis and opt.vis_mode != 'segm':
            dot_path = ''
            self.vis = Visual(mode=opt.vis_mode, save=True, svg=False, saved_dots=dot_path)
            self.vis.fit(self._embedded_feat, long_gt, 'gt_', reset=False)
            self.vis.color(long_rt, 'time_')
            self.vis.color(kmean.labels_, 'kmean')
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        self.pseudo_gt_with_bg[self.pseudo_gt_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(self.pseudo_gt_with_bg[video.global_range])

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)

    def _count_subact(self):
        self._subact_counter = np.zeros(self._K)
        for video in self._videos:
            self._subact_counter += video.a

    def generate_pi(self, pi, n_ins=0, n_del=0):
        output = pi.copy()
        for _ in range(n_del):
            n = len(output)
            idx = np.random.randint(n)
            output.pop(idx)

        for _ in range(n_ins):
            m = len(pi)
            val = np.random.randint(m)
            n = len(output)
            idx = np.random.randint(n)
            output.insert(idx, val)

        return output

    @timing
    def viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        max_score_list = []
        
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            if opt.bg:
                video.update_fg_mask()

            # for i in range(10):
            
            self.decoder.length_model = PoissonModel(self.mean_lengths)

            max_score, max_z, max_pi = self.video_decode(video, self.decoder)
            # print(video.shape)
            logger.debug('video length' + str(len(video._likelihood_grid)))
            max_score_list.append(max_score/len(video._likelihood_grid))
            
            if len(max_z) <= 0:
                continue

            self.pseudo_gt_with_bg[video.global_range] = max_z
            # self._z = np.asarray(alignment).copy()

                # self.train_classifier(video)

            self.buffer.add_sequence(max_z[video.fg_mask], max_pi, max_z[video.fg_mask])
            self.update_prior()
            self.update_mean_lengths()
            

            video._subact_count_update()
            video._z = np.asarray(max_z).copy()

            name = str(video.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
            np.savetxt(join(opt.output_dir, 'segmentation', name),
                    np.asarray(max_z), fmt='%d')

            
            
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
        self._count_subact()

        logger.debug('Q value' + str(np.mean(max_score_list)))
        logger.debug(str(self._subact_counter))

        # length_file = join(opt.output_dir, opt.subaction, 'mean_lengths.txt')
        # prior_file = join(opt.output_dir, opt.subaction, 'prior.txt')
        # np.savetxt(length_file, self.mean_lengths)
        # np.savetxt(prior_file, self.prior)

    def video_decode(self, video, decoder):
        max_score = -np.inf
        max_z = []
        max_pi = []
        pi = video._pi            
        for i in range(1):
            if i == 0:
                transcript = self.generate_pi(pi, n_ins=0, n_del=0)
            elif i <= 10:
                transcript = self.generate_pi(pi, n_ins=1, n_del=0)
            elif i <=20:
                transcript = self.generate_pi(pi, n_ins=0, n_del=1)
            elif i <=30:
                transcript = self.generate_pi(pi, n_ins=1, n_del=1)

            if np.sum(video.fg_mask):
                log_probs = video._likelihood_grid[video.fg_mask] - np.log(self.prior)
                log_probs = log_probs - np.max(log_probs) 
                decoder.grammar = SingleTranscriptGrammar(transcript, self._K)
                score, labels, segments = decoder.decode(log_probs)
                z = np.ones(video.n_frames, dtype=int) * -1
                z[video.fg_mask] = labels                
            else:
                z = np.ones(video.n_frames, dtype=int) * -1
                score = -np.inf
            # viterbi.calc(z)
            if score > max_score:
                max_score = score
                max_z = z
                max_pi = transcript

        return max_score, max_z, max_pi


    def without_temp_emed(self):
        logger.debug('No temporal embedding')
        self._embedded_feat = self._features.copy()

    @timing
    def accuracy_corpus(self, prefix=''):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        accuracy = Accuracy()
        f1_score = F1Score(K=self._K, n_videos=len(self._videos))
        long_gt = []
        long_pr = []
        long_rel_time = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video.gt)
            long_pr += list(video._z)
            try:
                long_rel_time += list(video.temp)
            except AttributeError:
                pass
                # logger.debug('no poses')
        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]

        old_mof, total_fr = accuracy.mof(old_gt2label=self._gt2label)
        self._gt2label = accuracy._gt2cluster
        self._label2gt = {}
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        logger.debug('%sAction: %s' % (prefix, self._subaction))
        logger.debug('%sMoF val: ' % prefix + str(acc_cur))
        logger.debug('%sprevious dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

        accuracy.mof_classes()
        accuracy.iou_classes()

        self.return_stat = accuracy.stat()

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(self._gt2label)
        if opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        for key, val in f1_score.stat().items():
            self.return_stat[key] = val

        for video in self._videos:
            video.segmentation[video.iter] = (video._z, self._label2gt)

        if opt.vis:
            ########################################################################
            # VISUALISATION

            if opt.vis_mode != 'segm':
                long_pr = [self._label2gt[i] for i in long_pr]

                if self.vis is None:
                    self.vis = Visual(mode=opt.vis_mode, save=True, reduce=None)
                    self.vis.fit(self._embedded_feat, long_pr, 'iter_%d' % self.iter)
                else:
                    reset = prefix == 'final'
                    self.vis.color(labels=long_pr, prefix='iter_%d' % self.iter, reset=reset)
            else:
                ####################################################################
                # visualisation of segmentation
                if prefix == 'final':
                    colors = {}
                    cmap = plt.get_cmap('tab20')
                    for label_idx, label in enumerate(np.unique(long_gt)):
                        if label == -1:
                            colors[label] = (0, 0, 0)
                        else:
                            # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                            colors[label] = cmap(label_idx / len(np.unique(long_gt)))

                    dir_check(os.path.join(opt.dataset_root, 'plots'))
                    dir_check(os.path.join(opt.dataset_root, 'plots', opt.subaction))
                    fold_path = os.path.join(opt.dataset_root, 'plots', opt.subaction, 'segmentation')
                    dir_check(fold_path)
                    for video in self._videos:
                        path = os.path.join(fold_path, video.name + '.png')
                        name = video.name.split('_')
                        name = '_'.join(name[-2:])
                        plot_segm(path, video.segmentation, colors, name=name)
                ####################################################################
            ####################################################################

        return accuracy.frames()

    def resume_segmentation(self):
        logger.debug('resume precomputed segmentation')
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()

    def save_embed_feat(self):
        dir_check(ops.join(opt.data, 'embed'))
        dir_check(ops.join(opt.data, 'embed', opt.subaction))
        for video in self._videos:
            video_features = self._embedded_feat[video.global_range]
            feat_name = opt.resume_str + '_%s' % video.name
            np.savetxt(ops.join(opt.data, 'embed', opt.subaction, feat_name), video_features)

    def update_mean_lengths(self):
        self.mean_lengths = np.zeros( (self._K), dtype=np.float32 )
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self._K), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 \
                else sum(self.mean_lengths) / sum(instances) for i in range(self._K) ] )

    def update_prior(self):
        # count labels
        self.prior = np.zeros((self._K), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self._K)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self._K for i in range(self._K) ] )

