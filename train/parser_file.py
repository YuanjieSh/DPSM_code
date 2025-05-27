
def initialize(parser):

  parser.add_argument('--seed', default = 123, type = int, help = 'seed for data splitting & model randomization')
  parser.add_argument('--arc', default = 'vgg', type = str, help = 'model architecture')
  parser.add_argument('--data', default = 'cifar100', type = str, help = 'data set')
  ## learning parameters
  parser.add_argument('--batch_size', default = 64, type = int, help = 'batch size')
  parser.add_argument('--num_epochs', default = 300, type = int, help = 'total number of epochs')
  parser.add_argument('--base_lr', default = 0.1, type = float, help = 'initial learning rate')
  parser.add_argument('--base_optimizer', default = 'SGD', choices = ['SGD', 'Adam'], help = 'which optimizer, SGD or Adam')
  parser.add_argument('--mu', default = 0.0, type = float, help = 'conformal loss weights (relative to cross entropy)')
  parser.add_argument('--train_alpha',default = 0.1, type = float, help = 'coverage level for training')
  parser.add_argument('--mu_size', default = 0.0, type = float, help = 'size loss parameter')
  parser.add_argument('--mu_class', default = 0.0, type = float, help = 'class loss parameter')
  parser.add_argument('--mu_lambda', default = 0.0, type = float, help = 'lambda loss parameter for CPL')
  parser.add_argument('--mu_qr', default = 10, type = float, help = 'QR loss parameter')
  parser.add_argument('--baseloss', default = 'CE', type = str, help = 'alternative loss in place of cross entropy')
  parser.add_argument('--test_alpha',default = 0.1, type = float, help = 'coverage level for training')
  #parser.add_argument('--mu_p', default = 0.0, type = float, help = 'pinnball loss parameter')

  ## number of data points
  parser.add_argument('--n_tr_samples', default = 450, type = int, help = 'number of data points used for training per class')
  parser.add_argument('--n_ho_samples', default = 80, type = int, help = 'number of data points hold out for early stoping per class')
  parser.add_argument('--n_test_samples', default = 50, type = int, help = 'number of data points used for testing after calibration per class')
  parser.add_argument('--n_cal_samples', default = 50, type = int, help = 'number of data points used for calibration per class')
  ## calibration and evaluations
  parser.add_argument('--cal_alpha',default = 0.1, type = float, help = 'coverage level for calibration')
  parser.add_argument('--return_scores', default = False, type = bool, help = 'return conformity scores on test dataset')
  parser.add_argument('--predict_prob_only', default = False, type = bool, help = 'return predicted probabilities on test dataset')
  parser.add_argument('--evaluation_condition', default = 'by_flag', type = str, help = 'condition criterion for evaluating coverage/prediction set size')
  parser.add_argument('--no_calib', default = False, type = bool, help = 'if no, do not calibrate/evaluate models directly on test dataset')
  parser.add_argument('--bs_times', default = 1, type = int, help = 'boostraping, 1 if no bootstrap, otherwise do bootstraps and evaluate on each subset')
  ## results saving
  parser.add_argument('--early_stopping',default = True, choices =[True, False], type = bool, help = 'use early stopping')
  parser.add_argument('--save_model',default = True, choices =[True, False], type = bool, help = 'save trained model')
  parser.add_argument('--save_checkpoint_period', default = 1, type = int, help = 'checkpoint save frequency')
  parser.add_argument('--save_pickle', default = False, type = str, help = 'save evaluation results in pickle')
  ## Imbalanced factor for the training and validation data
  parser.add_argument('--train_rho',default = 1.0, type = float, help = 'Training imbalanced factor')
  parser.add_argument('--val_rho',default = 1.0, type = float, help = 'Validation imbalanced factor')
  ## Train rule
  parser.add_argument('--train_rule', default='None', type=str, help='Training rule')
  ## For distribution matching loss
  parser.add_argument('--distribution', default = 'uniform', type = str, help = 'We need to match the CDF of the conformal distribution with which of the CDF distribution')
  parser.add_argument('--CDF_shift', default = 'KS', type = str, help = 'The CDF gap we want to minimize. Either KS:Kolmogorov-Smirnov or CM: Cramer-von Mises')
  parser.add_argument('--method', default = 'Conformal', type = str, help = 'Which method we want to apply')
  parser.add_argument('--cal_test_CP_score', default = 'APS', type = str, help = 'select one of the [APS, HPS]')

  ## Number of experiments
  parser.add_argument('-s', '--splits', default=10, type=int, help='Number of experiments to estimate mean set size and coverage')

  parser.add_argument('--marginal_calibration', default=True, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--class_wise_calibration', default=False, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--num_device', default=0, type=int, help='Which device will be used to run the code')
  parser.add_argument('--save_every', default=5, type=int, help='Which device will be used to run the code')

  parser.add_argument('--num_classes', default=100, type=int, help='number of classes for the data')
  parser.add_argument('--coverage_on_label', default=False, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--previous', default=False, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--finetune', default=False, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--finetune_epochs', default = 40, type = int, help = 'total number of epochs')
  parser.add_argument('--base_momentum', default = 0.9, type = float, help = 'total number of epochs')
  parser.add_argument('--load_weights', default=0, type=int, help='Number of experiments to estimate mean set size and coverage')
  parser.add_argument('--train_CP_score', default = 'HPS', type = str, help = 'select one of the [APS, HPS]')
  parser.add_argument('--finetune_momentum', default = 0.0, type = float, help = 'total number of epochs')
  parser.add_argument('--train_T', default=3.0, type=float, help='number of classes for the data')
  parser.add_argument('--sigmid_T', default=1.0, type=float, help='sigmoid temperature parameter')
  parser.add_argument('--cal_test_T', default=1.0, type=float, help='number of classes for the data')
  parser.add_argument('--base_lr_schedule', nargs='+', type = int, help='in what epochs we want to decay the learning rates')
  parser.add_argument('--finetune_lr_schedule', nargs='+', type = int, help='in what epochs we want to decay the learning rates')

  parser.add_argument('--finetune_batch_size', default = 128, type = int, help = 'batch size')
  parser.add_argument('--finetune_lr', default = 0.1, type = float, help = 'initial learning rate')
  parser.add_argument('--lr_qr', default = 0.01, type = float, help = 'learning rate only for the pinball loss optimization')
  parser.add_argument('--lr_h', default = 0.01, type = float, help = 'learning rate only for h in CPL')
  parser.add_argument('--lr_lamda', default = 0.01, type = float, help = 'learning rate only for lamda in CPL')
  parser.add_argument('--finetune_optimizer', default = 'SGD', choices = ['SGD', 'Adam'], help = 'which optimizer, SGD or Adam')
  parser.add_argument('--base_gamma', default = 0.1, type = float, help = 'initial learning rate')
  parser.add_argument('--base_weight_decay', default = 1e-4, type = float, help = 'initial learning rate')

  parser.add_argument('--finetune_gamma', default = 0.1, type = float, help = 'initial learning rate')
  parser.add_argument('--finetune_weight_decay', default = 1e-4, type = float, help = 'initial learning rate')
  parser.add_argument('--qr_decay_a', default = 2, type = float, help = 'QR learning rate decay parameter a')
  parser.add_argument('--qr_decay_b', default = 0.5, type = float, help = 'QR learning rate decay parameter b')

  parser.add_argument('--finetune_CE', default = False, type = int, help = 'initial learning rate')
  
  parser.add_argument('--classwise_epochs', default = 10, type = int, help = 'total number of epochs')
  parser.add_argument('--classwise_training', default = False, type = int, help = 'total number of epochs')
  
  parser.add_argument('--APS_training', default = False, type = int, help = 'total number of epochs')
  parser.add_argument('--APS_epochs', default = 10, type = int, help = 'total number of epochs')

  parser.add_argument('--RAPS_training', default = False, type = int, help = 'total number of epochs')
  parser.add_argument('--RAPS_epochs', default = 10, type = int, help = 'total number of epochs')


  return parser

