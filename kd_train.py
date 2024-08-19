import args as program_args
import training        

if __name__ == "__main__":
    parser = program_args.get_arg_parser()
    args = parser.parse_args()
    training.main(args)

