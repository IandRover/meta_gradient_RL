import torch   

"""Compute df/dtheta"""
def compute_df_dtheta(pi_, values_, return_, actor_network, value_network, gamma, c=0.5):

    theta1 = torch.autograd.grad(pi_, actor_network.parameters())
    theta1 = [item.view(-1) for item in theta1]
    theta1 = torch.cat(theta1)

    theta2 = torch.autograd.grad(values_, value_network.parameters())
    theta2 = [item.view(-1) for item in theta2]
    theta2 = torch.cat(theta2)

    theta_sum = theta1

    g_dgamma = torch.autograd.grad(return_, gamma)

    return g_dgamma[0], theta1, theta2

def compute_dfpron_dthetapron(pi_, values_, actor_network, value_network):

    theta1 = torch.autograd.grad(pi_, actor_network.parameters())
    theta1 = [item.view(-1) for item in theta1]
    theta1 = torch.cat(theta1)
    
    theta2 = torch.autograd.grad(values_, value_network.parameters())
    theta2 = [item.view(-1) for item in theta2]
    theta2 = torch.cat(theta2)
    
    return theta1, theta2
