"""
多材质定量：NET2 逐场景评估，按材质分组统计
"""
import os, json, sys, time, numpy as np
import torch, torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = r'D:\LZ\zhoukai_paper'
LZ_DIR = os.path.join(PROJECT_ROOT, 'lz')
sys.path.insert(0, LZ_DIR)

from Snet3_2 import PolarDenoiseNet  # NET4, but we'll use NET2 architecture
# Actually NET2 uses S_Net3_1.py with a different class
import importlib.util
net2_spec = importlib.util.spec_from_file_location('NET2', os.path.join(LZ_DIR, '3_1-9-35', 'S_Net3_1.py'))
net2_mod = importlib.util.module_from_spec(net2_spec)
net2_spec.loader.exec_module(net2_mod)
PolarDenoiseNet_NET2 = net2_mod.PolarDenoiseNet

DEVICE = torch.device('cuda')
JSON_PATH = os.path.join(PROJECT_ROOT, r'2\new\data2.1\dataset_info.json')
OLD_PREFIX, NEW_PREFIX = r'G:\zzz\2', os.path.join(PROJECT_ROOT, '2')

# 材质映射
SCENE_MATERIAL = {
    '1':'Shiny Metal','2':'Shiny Metal','3':'Shiny Metal','4':'Shiny Metal',
    '5':'Shiny Metal','6':'Shiny Metal','7':'Shiny Metal','8':'Shiny Metal',
    '9':'Shiny Metal','10':'Dark Metal','11':'Mixed Metal','12':'Mixed Metal',
    '13':'Shiny Metal','14':'Rubber','15':'Dark Metal','16':'Plastic',
    '17':'Cardboard','18':'Plastic','19':'Ceramic','20':'Dark Metal','21':'PCB',
}

def remap(p):
    return p.replace(OLD_PREFIX, NEW_PREFIX, 1) if p.startswith(OLD_PREFIX) else p

def read_img(p):
    return np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0

def gk(size, dev='cpu'):
    s = size/3.0; ax = torch.arange(size).float().to(dev)-(size-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx**2+yy**2)/(2*s**2)); return k/k.max()

def compute_aop(I0,I45,I90,I135):
    return 0.5*torch.atan2(I45-I135, I0-I90)*180/np.pi

def psnr(p,t,m=1.0):
    mse=F.mse_loss(p.squeeze(),t.squeeze()); return float('inf') if mse==0 else 10*torch.log10(m**2/mse).item()

def ssim(p,t):
    dev = p.device; C1,C2=0.01**2,0.03**2
    p=p.squeeze().unsqueeze(0).unsqueeze(0); t=t.squeeze().unsqueeze(0).unsqueeze(0)
    k=torch.ones(1,1,11,11,device=dev)/121.
    mu1=F.conv2d(p,k,padding=5); mu2=F.conv2d(t,k,padding=5)
    s1=F.conv2d(p*p,k,padding=5)-mu1**2; s2=F.conv2d(t*t,k,padding=5)-mu2**2
    s12=F.conv2d(p*t,k,padding=5)-mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2)/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()

def predict(model, noisy, dev):
    model.eval()
    with torch.no_grad():
        C,H,W = noisy.shape
        acc = {k:torch.zeros(H,W,device=dev) for k in ['I0','I45','I90','I135','dolp']}
        ws = torch.zeros(H,W,device=dev); g = gk(128,dev)
        tops = list(range(0,H-128+1,64)); tops.append(H-128) if tops[-1]!=H-128 else None
        lefts = list(range(0,W-128+1,64)); lefts.append(W-128) if lefts[-1]!=W-128 else None
        for t in tops:
            for l in lefts:
                p = noisy[:,t:t+128,l:l+128].unsqueeze(0).to(dev)
                o = model(p[:,0:1],p[:,1:2],p[:,2:3],p[:,3:4])
                for k in acc: acc[k][t:t+128,l:l+128] += o[k][0,0]*g
                ws[t:t+128,l:l+128] += g
        ws=torch.clamp(ws,min=1e-8)
        return {k:torch.clamp(acc[k]/ws,0,1) for k in acc}

print(f"GPU: {torch.cuda.get_device_name(0)}")
model = PolarDenoiseNet_NET2(base_channels=32).to(DEVICE)
ckpt = torch.load(os.path.join(LZ_DIR,'3_1-9-35','best_model.pth'), map_location=DEVICE)
if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict'])
else: model.load_state_dict(ckpt)
model.eval()
print("Model loaded: NET2 (DynamicConv + PolarAttention)")

with open(JSON_PATH) as f: scenes = json.load(f)

samples = []
for sd in scenes:
    gt, noise = sd['gt_images'], sd['noise_images']
    for level, paths in noise.items():
        samples.append({
            'scene': sd['scene'], 'noise': level,
            'noise_paths': [remap(paths[k]) for k in ['I0','I45','I90','I135']],
            'gt_paths': [remap(gt[k]) for k in ['I0','I45','I90','I135','dolp']],
        })

print(f"Total samples: {len(samples)}")

# 按材质分组收集
mat_results = {}
t0 = time.time()

for idx, s in enumerate(samples):
    try:
        noisy = np.stack([read_img(p) for p in s['noise_paths']], axis=0)
        noisy_t = torch.from_numpy(noisy).float().to(DEVICE)
        gts = [torch.from_numpy(read_img(p)).float().to(DEVICE) for p in s['gt_paths']]
        gt_I0,gt_I45,gt_I90,gt_I135,gt_dolp = gts

        pred = predict(model, noisy_t, DEVICE)
        pI0,pI45,pI90,pI135,pDolp = pred['I0'],pred['I45'],pred['I90'],pred['I135'],pred['dolp']

        mat = SCENE_MATERIAL.get(s['scene'], 'Unknown')
        if mat not in mat_results:
            mat_results[mat] = {'dolp_psnr':[],'dolp_ssim':[],'aop_rmse':[],'count':0}

        mat_results[mat]['dolp_psnr'].append(psnr(pDolp, gt_dolp))
        mat_results[mat]['dolp_ssim'].append(ssim(pDolp, gt_dolp))
        mat_results[mat]['aop_rmse'].append(float(torch.sqrt(torch.mean(
            (compute_aop(pI0,pI45,pI90,pI135)-compute_aop(gt_I0,gt_I45,gt_I90,gt_I135))**2)).item()))
        mat_results[mat]['count'] += 1
    except Exception as e:
        pass

    if (idx+1)%20==0:
        print(f"  [{idx+1}/{len(samples)}] elapsed={time.time()-t0:.0f}s")

print(f"\nDone in {time.time()-t0:.0f}s\n")
print("="*70)
print("           Multi-Material Results (NET2)")
print("="*70)
print(f"{'Material':<20} {'Samples':>8} {'DoLP PSNR':>12} {'DoLP SSIM':>12} {'AoP RMSE':>12}")
print("-"*70)

for mat in sorted(mat_results.keys()):
    r = mat_results[mat]
    print(f"{mat:<20} {r['count']:>8} {np.mean(r['dolp_psnr']):>12.2f} "
          f"{np.mean(r['dolp_ssim']):>12.4f} {np.mean(r['aop_rmse']):>12.2f}")

# 合并大类
print("\n--- Coarse Groups ---")
coarse = {
    'Metal (shiny)': ['Shiny Metal'],
    'Metal (dark)': ['Dark Metal','Mixed Metal'],
    'Plastic/Rubber': ['Rubber','Plastic'],
    'Other': ['Cardboard','Ceramic','PCB'],
}
for cg, mats in coarse.items():
    all_d, all_s, all_a = [], [], []
    for m in mats:
        if m in mat_results:
            all_d.extend(mat_results[m]['dolp_psnr'])
            all_s.extend(mat_results[m]['dolp_ssim'])
            all_a.extend(mat_results[m]['aop_rmse'])
    if all_d:
        print(f"{cg:<20} {len(all_d):>8} {np.mean(all_d):>12.2f} "
              f"{np.mean(all_s):>12.4f} {np.mean(all_a):>12.2f}")

# 保存
out = {'per_material': {}, 'coarse': {}}
for mat, r in mat_results.items():
    out['per_material'][mat] = {k: (np.mean(v) if isinstance(v,list) else v) for k,v in r.items()}
with open(os.path.join(LZ_DIR,'exp_all_results','material_results.json'),'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to exp_all_results/material_results.json")
