import torch
import triton

torch.manual_seed(0)
torch.set_printoptions(precision=4)

x = torch.autograd.Variable(torch.randn(16, 16, 8, 8).cuda(), requires_grad=True)
bias = torch.autograd.Variable(torch.randn(16).cuda(), requires_grad=True)
w = torch.autograd.Variable(torch.randn(16, 3, 3, 16).cuda(), requires_grad=True)
cuw = torch.autograd.Variable(w.permute(3,0,1,2).cuda(), requires_grad=True)
y_target = torch.autograd.Variable(torch.randn(16, 16, 3, 3).cuda(), requires_grad=True)

def run(x, w, conv):
  y = conv(x, w)
  loss = (y - y_target).norm(2)
  loss.backward()
  return loss, y.clone(), x.grad.clone(), w.grad.clone(), bias.grad.clone()

ttyloss, tty, ttdx, ttdw, ttbias = run(x, w, lambda x, w: triton.ConvFunction.apply(x, w, bias, (2,2), (0,0)))
x.grad.zero_()
w.grad.zero_()
bias.grad.zero_()
culoss, cuy, cudx, cudw, cubias = run(x, cuw, lambda x, w: torch.nn.functional.conv2d(x, w, bias=bias, stride=2, padding=0))

print(ttdw.permute(3,0,1,2)[0,0,:,:])
print(cudw[0,0,:,:])
print((tty - cuy).norm(2))
print((ttdx - cudx).norm(2))
print((ttdw.permute(3,0,1,2) - cudw).norm(2))
print((ttbias - cubias).norm(2))
