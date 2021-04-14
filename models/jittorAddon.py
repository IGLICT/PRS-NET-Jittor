import jittor as jt
from jittor.contrib import concat 
from jittor import init, Module
from jittor.misc import _triple
import math

class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw, Kd = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw, Kd], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if self.groups == 1:
            N,C,H,W,D = x.shape
            Kh, Kw, Kd = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            od = (D+self.padding[2]*2-Kd*self.dilation[2]+self.dilation[2]-1)//self.stride[2]+1
            xx = x.reindex([N,self.out_channels,C,oh,ow,od,Kh,Kw,Kd], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i4*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
                f'i5*{self.stride[2]}-{self.padding[2]}+i8*{self.dilation[2]}', # Did+KDid
            ])
            ww = self.weight.broadcast(xx.shape, [0,3,4,5])
            yy = xx*ww
            y = yy.sum([2,6,7,8]) # Kc, Kh, Kw, Kd
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3,4])
                y = y + b
            return y
        else:
            N,C,H,W,D = x.shape
            Kh, Kw, Kd = self.kernel_size
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            od = (D+self.padding[2]*2-Kd*self.dilation[2]+self.dilation[2]-1)//self.stride[2]+1
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,od,Kh,Kw,Kd], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i7*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i8*{self.dilation[1]}', # Wid+KWid
                f'i6*{self.stride[2]}-{self.padding[2]}+i9*{self.dilation[2]}', # Did+KDid
            ])
            # w: [oc, CpG, Kh, Kw, Kd]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, od, Kh, Kw, Kd], [
                f'i1*{oc//G}+i2',
                'i3',
                'i7',
                'i8',
                'i9'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            yy = xx*ww
            y = yy.reindex_reduce('add', [N, oc, oh, ow, od], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5',
                'i6'
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3,4])
                y = y + b
            return y 



class Pool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W,D = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding*2-self.kernel_size)//self.stride+1
            w = (W+self.padding*2-self.kernel_size)//self.stride+1
            d = (D+self.padding*2-self.kernel_size)//self.stride+1
        else:
            h = (H+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            w = (W+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            d = (D+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1

        if self.op in ['maximum', 'minimum', 'mean']:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size*self.kernel_size*self.kernel_size};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3)* (k4_ - k4);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''{{
                int k4 = i4*{self.stride}-{self.padding};
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k4_ = min(k4 + {self.kernel_size}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                @out(i0, i1, i2, i3, i4) = init_{self.op}(out_type);
                {count}
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = {self.op}(out_type, @out(i0, i1, i2, i3, i4), @in0(i0, i1, p, q, r));
            }}'''
            backward_body = f'''{{
                int k4 = i4*{self.stride}-{self.padding};
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k4_ = min(k4 + {self.kernel_size}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) 
                        for (int r = k4; r < k4_ && bo; ++r) {{
                        {"atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)/count);"
                            if self.op == "mean" else
                        f"""if (@pout(i0,i1,i2,i3,i4) == @in0(i0,i1,p,q,r)) {{
                            atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)),
                            bo=0;
                        }}"""}
                    }}
            }}'''
            out = jt.code([N,C,h,w,d], x.dtype, [x],
                cuda_header="""
                    #include <ops/binary_op_defs.h>
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int res_x = (in0_shape4 - 1) / blockDim.x + 1;
                        int res_y = (in0_shape3 - 1) / blockDim.y + 1;
                        int res_z = (in0_shape2 - 1) / blockDim.z + 1;

                        int idx4 = blockIdx.x / (res_y * res_z);
                        int idx3 = (blockIdx.x - idx4 * res_y * res_z) / res_z;
                        int idx2 = blockIdx.x - idx4 * res_y * res_z - idx3 * res_y;


                        int p4 = threadIdx.x + idx4 * blockDim.x;
                        int s4 = blockDim.x * res_x;

                        int p3 = threadIdx.y + idx3 * blockDim.y;
                        int s3 = blockDim.y * res_y;

                        int p2 = threadIdx.z + idx2 * blockDim.z;
                        int s2 = blockDim.z * res_z;

                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {forward_body}
                    }}

                    int tx = min(1024, out_shape4);
                    int ty = min(1024 / tx, out_shape3);
                    int tz = min(1024 / tx / ty, out_shape2);


                    int res_x = (out_shape4 - 1) / tx + 1;
                    int res_y = (out_shape3 - 1) / ty + 1;
                    int res_z = (out_shape2 - 1) / tz + 1;

                    

                    int bx = res_x * res_y * res_z;
                    int by = out_shape1;
                    int bz = out_shape0;

                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC


                        int res_x = (in0_shape4 - 1) / blockDim.x + 1;
                        int res_y = (in0_shape3 - 1) / blockDim.y + 1;
                        int res_z = (in0_shape2 - 1) / blockDim.z + 1;

                        int idx4 = blockIdx.x / (res_y * res_z);
                        int idx3 = (blockIdx.x - idx4 * res_y * res_z) / res_z;
                        int idx2 = blockIdx.x - idx4 * res_y * res_z - idx3 * res_y;


                        int p4 = threadIdx.x + idx4 * blockDim.x;
                        int s4 = blockDim.x * res_x;

                        int p3 = threadIdx.y + idx3 * blockDim.y;
                        int s3 = blockDim.y * res_y;

                        int p2 = threadIdx.z + idx2 * blockDim.z;
                        int s2 = blockDim.z * res_z;

                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < pout_shape4; i4 += s4)
                            for (int i3 = p3; i3 < pout_shape3; i3 += s3)
                                for (int i2 = p2; i2 < pout_shape2; i2 += s2)
                                    {backward_body}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);

                    int tx = min(1024, pout_shape4);
                    int ty = min(1024 / tx, pout_shape3);
                    int tz = min(1024 / tx / ty, pout_shape2);

                    int res_x = (pout_shape4 - 1) / tx + 1;
                    int res_y = (pout_shape3 - 1) / ty + 1;
                    int res_z = (pout_shape2 - 1) / tz + 1;

                    int bx = res_x * res_y * res_z;

                    int by = pout_shape1;

                    int bz = pout_shape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty, tz);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
                cpu_header='#include <ops/binary_op_defs.h>',
                cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                    for (int i4=0; i4<out_shape4; i4++)
                        {forward_body}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b
                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
                    for (int i3=0; i3<pout_shape3; i3++)
                    for (int i4=0; i4<pout_shape4; i4++)
                        {backward_body}
                '''])
            return out
        else:
            # TODO: backward 
            xx = x.reindex([N,C,h,w,self.kernel_size,self.kernel_size], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride}-{self.padding}+i4", # Hid
                f"i3*{self.stride}-{self.padding}+i5", # Wid
            ])
            return xx.reduce(self.op, [4,5])
