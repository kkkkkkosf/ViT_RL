import torch

grad_map = {}

def hook_grad(key, value):
    grad_map[key] = value

def GAE(reward, value, gamma, lam):
    adv = torch.FloatTensor(reward.shape[0], 1).cuda()
    delta = torch.FloatTensor(reward.shape[0], 1).cuda()
    pre_value, pre_adv = 0, 0
    for i in reversed(range(reward.shape[0])):
        delta[i] = reward[i] + gamma * pre_value - value[i]
        adv[i] = delta[i] + gamma * lam * pre_adv
        pre_adv = adv[i, 0]
        pre_value = value[i, 0]

    returns = value + adv
    adv = (adv - adv.mean()) / adv.std()
    return adv, returns

def PPO_step(policy_net, value_net, policy_optim, value_optim, state, action, returns, advantage,
             old_log_prob, epsilon, l2_reg, soft_value):
    value_optim.zero_grad()
    value_o = value_net(action.detach())
    v_loss = (value_o - returns.detach()).pow(2).mean()
    for param in value_net.parameters():
        v_loss += param.pow(2).sum() * l2_reg
    v_loss.register_hook(lambda grad: hook_grad("v_loss", grad))
    v_loss.backward()
    value_optim.step()
    policy_optim.zero_grad()
    log_prob = policy_net.module.get_log_prob(state.detach(), soft_value)
    ratio = torch.exp(log_prob - old_log_prob.detach())
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    p_loss = -torch.min(surr1, surr2).mean()
    p_loss.register_hook(lambda grad: hook_grad("p_loss", grad))
    p_loss *= 1000000
    p_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    policy_optim.step()
    return v_loss, p_loss

