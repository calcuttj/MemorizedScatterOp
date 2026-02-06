from argparse import ArgumentParser as ap, RawTextHelpFormatter as RTHF

import torch
import torch.profiler as profiler

if __name__ == '__main__':

    parser = ap(formatter_class=RTHF)
    parser.add_argument('--library', type=str, required=True)
    parser.add_argument('--operation', type=str, default='sum')
    parser.add_argument(
        '--indices',
        help='''
Either the length of random indices tensor 
OR
file:tensor:loc --
    file: File holding the indices
    tensor: name of tensor in file
    loc:loc in tensor (cells are usually NCells x 3planes)''',
        
    )
    parser.add_argument(
        '--shape',
        help='Shape of the frame tensor',
        default=[600, 2560],
        nargs=2,
        type=int,
    )
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if ':' in args.indices:
        f, t, i = args.indices.split(':')
        f = torch.load(f)
        indices = f[t][int(i)].clone().to(args.device)
        print(indices)
    else:
        ind_len = int(args.indices)
        indices = torch.randint(args.shape[1], (ind_len,), device=args.device)
        print(indices)

    #Load library + inst
    torch.ops.load_library(args.library)
    op = torch.classes.my_ops.MyScatterOp(indices)
    # op.to(args.device)

    frame = torch.zeros(args.shape, device=args.device)
    print(frame)

    cells = torch.randn(args.shape[0], indices.shape[0], device=args.device)
    print(cells)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profiler.profile(

            schedule=torch.profiler.schedule(wait=1, warmup=1, active=6, repeat=1),

            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/scatter_profile'),
            record_shapes=True,
            with_stack=True, profile_memory=True) as prof:

            for i in range(10):
                frame += op.forward(cells, frame.shape, args.operation)
                prof.step()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print(frame)