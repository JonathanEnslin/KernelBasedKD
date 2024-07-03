'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
from torch.nn.functional import softmax

def softmax_with_temperature(logits, temperature) -> torch.float32:
    return torch.exp(logits/temperature) / torch.sum(torch.exp(logits/temperature))

def batch_softmax_with_temperature(batch_logits, temperature) -> torch.tensor:
    return softmax(batch_logits/temperature,dim=1)

if __name__ == '__main__':
    logits = torch.tensor([1., 2., 3.])
    batch_logits = torch.tensor([[1., 2., 3.],[1., 2., 3.],[1., 2., 3.],[1., 5., 3.]])
    print('logits value: {} logits exponetialvalue type: {}'.format(logits,logits.dtype))
    logits_exp = torch.exp(logits)
    print('logits exponetial value: {} logits exponetial value type: {}'.format(logits_exp, logits_exp.dtype))

    #Larger values of temperature leads to a smoother distibution 
    TEMPERATURES = [1.,5.,7.,10.,100., 10000]
    for temperature in TEMPERATURES:
        result = softmax_with_temperature(logits, temperature)
        print('softmax value: {} softmax value type: {} temperature: {}'.format(result, result.dtype, temperature))


    result = batch_softmax_with_temperature(batch_logits, 1)
    print('softmax value for batch: {} softmax value type for batch: {} temperature: {}'.format(result, result.dtype, 1))


    
