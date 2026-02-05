import torch
import torch.autograd.profiler as profiler
from argparse import ArgumentParser as ap

def run_gradcheck(func, inputs):
    print("Running gradcheck...")
    # eps: step size for finite difference
    # atol: absolute tolerance
    test_passed = torch.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-4)
    
    if test_passed:
        print("✅ SUCCESS: Manual backward pass matches numerical gradients!")
    else:
        print("❌ FAILED: Gradient mismatch detected.")

def verify_spatial_logic(args):
    # Use small dimensions so gradcheck finishes quickly
    # Use double precision (float64) for numerical stability in testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    # 1. Setup inputs with requires_grad=True
    input_tensor = torch.randn(*args.inshape, device=device, dtype=dtype, requires_grad=True)
    
    last_dim_size = input_tensor.shape[-1]
    print(last_dim_size)
    ind = torch.randint(output_tensor.shape[-1], (input_tensor.shape[-1],)).to(device)
    
    op = torch.classes.my_ops.MyScatterOp(ind)
    
    # 3. Define a wrapper for gradcheck
    # gradcheck expects a function and a tuple of inputs
    def func(inp):
        output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
        shape = output_tensor.shape
        output_tensor += op.forward(
            inp, shape
        )
        return output_tensor

    run_gradcheck(func, (input_tensor,))

def chunked_verify_spatial_logic(args):
    # Use small dimensions so gradcheck finishes quickly
    # Use double precision (float64) for numerical stability in testing
    if (args.inshape[0] != args.outshape[0]):
        raise RuntimeError('in and out shapes need to match on first dims')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    chunk = args.chunk
    # 1. Setup inputs with requires_grad=True
    input_tensor = torch.randn(*args.inshape, device=device, dtype=dtype, requires_grad=True)
    last_dim_size = input_tensor.shape[-1]
    print(last_dim_size)
    ind = torch.randint(output_tensor.shape[-1], (input_tensor.shape[-1],)).to(device)
    
    op = torch.classes.my_ops.MyScatterOp(ind)
    
    # 3. Define a wrapper for gradcheck
    # gradcheck expects a function and a tuple of inputs
    def func(inp):
        output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
        for i in range(0, inp.shape[0], chunk):
            output_tensor[i:i+chunk] += op.forward(inp[i:i+chunk], output_tensor[i:i+chunk].shape)

        return output_tensor

    run_gradcheck(func, (input_tensor,))

def verify_6planes_chunked(args):
    # Use small dimensions so gradcheck finishes quickly
    # Use double precision (float64) for numerical stability in testing
    if (args.inshape[0] != args.outshape[0]):
        raise RuntimeError('in and out shapes need to match on first dims')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    chunk = args.chunk

    input_tensors = [
        torch.randn(*args.inshape, device=device, dtype=dtype, requires_grad=True)
        for i in range(6)
    ]
    inds = [
        torch.randint(args.outshape[-1], (args.inshape[-1],)).to(device)
        for i in range(6)
    ]
    
    ops = [torch.classes.my_ops.MyScatterOp(ind) for ind in inds]
    
    # 3. Define a wrapper for gradcheck
    # gradcheck expects a function and a tuple of inputs 
    def func(*inputs):
        output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
        for i in range(0, output_tensor.shape[0], chunk):
            for j in range(len(inputs)):
                output_tensor[i:i+chunk] += ops[j].forward(inputs[j][i:i+chunk], output_tensor[i:i+chunk].shape)

        return output_tensor

    run_gradcheck(func, tuple(input_tensors))

def profile(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=(torch.bfloat16 if args.bfloat16 else torch.float32)
    input_tensor = torch.randn(*args.inshape, device=device, dtype=dtype, requires_grad=True)
    output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
    target_shape = output_tensor.shape
    last_dim_size = output_tensor.shape[-1]
    ind = torch.randint(output_tensor.shape[-1], (input_tensor.shape[-1],)).to(device)
    
    op = torch.classes.my_ops.MyScatterOp(ind)
    torch.cuda.synchronize()
    print('Calling model forward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        output_tensor += op.forward(input_tensor, output_tensor.shape)
        torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')


    print('Summing')
    loss = output_tensor.sum()
    torch.cuda.synchronize()
    print('Calling backward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        loss.backward()
        torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

    print('Calling model without grad')
    with torch.no_grad():
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            y = op.forward(input_tensor, target_shape)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

    print(f'Running forward + backward {args.n} times')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        for i in range(args.n):
            y = op.forward(input_tensor, target_shape)
            l = y.sum()
            l.backward()
    torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

def call_chunked(output_tensor, inputs, ops, chunk):
    # output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
    for i in range(0, output_tensor.shape[0], chunk):
        for j in range(len(inputs)):
            output_tensor[i:i+chunk] += ops[j].forward(inputs[j][i:i+chunk], output_tensor[i:i+chunk].shape)
    return output_tensor

def profile_6planes_chunked(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=(torch.bfloat16 if args.bfloat16 else torch.float32)
    
    output_tensor = torch.zeros(*args.outshape, device=device, dtype=dtype)
    
    input_tensors = [
        torch.randn(*args.inshape, device=device, dtype=dtype, requires_grad=True)
        for i in range(6)
    ]
    inds = [
        torch.randint(args.outshape[-1], (args.inshape[-1],)).to(device)
        for i in range(6)
    ]
    
    ops = [torch.classes.my_ops.MyScatterOp(ind) for ind in inds]
    
    torch.cuda.synchronize()
    print('Calling model forward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        output_tensor = call_chunked(output_tensor, input_tensors, ops, args.chunk)
        torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')


    print('Summing')
    loss = output_tensor.sum()
    torch.cuda.synchronize()
    print('Calling backward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        loss.backward()
        torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

    print('Calling model without grad')
    with torch.no_grad():
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            output_tensor = call_chunked(output_tensor, input_tensors, ops, args.chunk)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('--library', type=str, help='Location of library on disk', required=True)
    subparser = parser.add_subparsers(dest='command')
    verify_parser = subparser.add_parser('verify')
    verify_parser.add_argument('--inshape', type=int, nargs='+', default=[1,2], help='shape of input')
    verify_parser.add_argument('--outshape', type=int, nargs='+', default=[1,10], help='shape of output')

    chunked_verify_parser = subparser.add_parser('chunked_verify')
    chunked_verify_parser.add_argument('--inshape', type=int, nargs='+', default=[1,2], help='shape of input')
    chunked_verify_parser.add_argument('--outshape', type=int, nargs='+', default=[1,10], help='shape of output')
    chunked_verify_parser.add_argument('--chunk', type=int, default=1)
    
    profile_parser = subparser.add_parser('profile')
    profile_parser.add_argument('--inshape', type=int, nargs='+', default=[1,2], help='shape of input')
    profile_parser.add_argument('--outshape', type=int, nargs='+', default=[1,10], help='shape of output')
    profile_parser.add_argument('--bfloat16', action='store_true', help='Use bfloat16 dtype')
    profile_parser.add_argument('-n', type=int, default=0, help='Number of times to repeat fwd+bwd runs')

    verify_6planes_parser = subparser.add_parser('verify_6planes_chunked')
    verify_6planes_parser.add_argument('--inshape', type=int, nargs='+', default=[1,2], help='shape of input')
    verify_6planes_parser.add_argument('--outshape', type=int, nargs='+', default=[1,10], help='shape of output')
    verify_6planes_parser.add_argument('--chunk', type=int, default=1)

    profile_6planes_parser = subparser.add_parser('profile_6planes_chunked')
    profile_6planes_parser.add_argument('--inshape', type=int, nargs='+', default=[1,2], help='shape of input')
    profile_6planes_parser.add_argument('--outshape', type=int, nargs='+', default=[1,10], help='shape of output')
    profile_6planes_parser.add_argument('--chunk', type=int, default=1)
    profile_6planes_parser.add_argument('--bfloat16', action='store_true', help='Use bfloat16 dtype')
    profile_6planes_parser.add_argument('-n', type=int, default=0, help='Number of times to repeat fwd+bwd runs')

    args = parser.parse_args()
    
    torch.ops.load_library(args.library)

    print(args.command)
    if args.command == 'verify':
        verify_spatial_logic(args)
    if args.command == 'chunked_verify':
        chunked_verify_spatial_logic(args)
    if args.command == 'profile':
        profile(args)
    if args.command == 'verify_6planes_chunked':
        verify_6planes_chunked(args)
    if args.command == 'profile_6planes_chunked':
        profile_6planes_chunked(args)