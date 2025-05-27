  def _run_batch_conformal(self, source, targets): #Romano's paper
    self.optimizer.zero_grad()
    output = self.model(source)
    
    """Add a temperature scaling T=1 in order to smooth the probability. Without scaling, the model gives very high probability for prediction that consists some noise.
    """
    Temp = self.ags.train_T
    output /= Temp
    
    ind1, ind2 = train_test_split(torch.arange(len(output)), train_size=0.8, random_state=1111)
    
    loss1 = self.loss_fn(output[ind1], targets[ind1])

    if self.ags.train_CP_score == 'APS':
      scores = find_scores_APS(self.soft_proba(output[ind2]), targets[ind2], device = self.local_rank)
    elif self.ags.train_CP_score == 'HPS':
      scores = find_scores_HPS(self.soft_proba(output[ind2]), targets[ind2], device = self.local_rank)
      
    loss2 = self.cdf_gap(scores, device = self.local_rank)
    loss = loss1 + self.mu * loss2
    loss.backward()
    self.optimizer.step()
    
    self.scores_return  = scores
    
    if self.local_rank == 0:  
      q_data1 = torch.quantile(scores, q = 1-self.train_alpha).detach().item()
      self.data_taus.append(q_data1)
      
    return loss1, loss2
  
  
  
  def _run_batch_pinball_class(self, source, targets):
    self.optimizer.zero_grad()
    output = self.model(source)
    ind1, ind2 = train_test_split(torch.arange(len(output)), train_size=0.499, random_state=1111)
    
    loss1 = self.loss_fn(output, targets)
    scores = find_scores_HPS(self.soft_proba(output), targets)

    loss3, tau = self.pinball_loss(scores[ind1], targets[ind1], device = self.local_rank)
    if self.local_rank == 0:  
      self.taus.append(tau.tolist())
    proba = self.soft_proba(output[ind2])
    size_loss, class_size_loss = Estimate_size_loss(proba, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = 1.0, K = 1.0)         
    #loss2 = torch.log(class_size_loss + size_loss * self.mu_size)
    loss2 = class_size_loss + size_loss * self.mu_size
    
    loss = loss1 + loss2 + self.mu_p * loss3
    loss.backward()
    self.optimizer.step()
    for p in self.pinball_loss.parameters():
      p.data.clamp_(0.00001, 0.9999999)
    return loss1, loss2, loss3