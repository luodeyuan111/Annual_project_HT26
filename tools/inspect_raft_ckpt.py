import torch
p='models/raft/raft-things.pth'
ckpt=torch.load(p,map_location='cpu')
print('TYPE:',type(ckpt))
if isinstance(ckpt, dict):
    print('TOP_KEYS_COUNT', len(ckpt))
    for k in list(ckpt.keys())[:20]:
        print('TOPKEY:', k, type(ckpt[k]))
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        print('\nSTATE_DICT_KEYS', len(sd))
        for k in list(sd.keys())[:60]:
            print('SD:', k)
    else:
        print('\nRAW_STATE_DICT_KEYS', len(ckpt))
        for k in list(ckpt.keys())[:60]:
            print('CK:', k)
else:
    print('Checkpoint is not a dict, repr:', repr(ckpt)[:200])
