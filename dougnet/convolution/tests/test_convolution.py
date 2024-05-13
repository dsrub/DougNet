import pytest


N = 1_000
C_in, C_out = 64, 128
H, W = 30, 30
H_K, W_K = 3, 3

V = np.random.randn(N, C_in, H, W)#.astype(np.float32)
K = np.random.randn(C_out, C_in, H_K, W_K)#.astype(np.float32)
b = np.random.randn(C_out,)#.astype(np.float32)

V_tensor = torch.tensor(V)
K_tensor = torch.tensor(K)
b_tensor = torch.tensor(b)


pad=3
stride = 2
dilate=3

conv = conv2d(V, K, b, pad=pad, stride=stride, dilate=dilate, method="gemm")
conv_torch = nn.functional.conv2d(V_tensor, K_tensor, b_tensor, stride=stride, padding=pad, dilation=dilate).numpy()

print(np.linalg.norm(conv - conv_torch))



test
- pad
- stride
- dilate
- dtype
- V_size
- K_size
- b_size
- method




@pytest.mark.parametrize("a", [1, 2, 3], ids=["doug", "me", "you"])
@pytest.mark.parametrize("b", [4, 5], ids=["him", "5"])
@pytest.mark.parametrize("c", [6, 7], ids=["6", "her"])
def tests(a, b, c):
    assert a==b