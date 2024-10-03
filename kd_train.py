import args as program_args

if __name__ == "__main__":
    parser = program_args.get_arg_parser()
    args = parser.parse_args()
    import training        
    training.main(args)

