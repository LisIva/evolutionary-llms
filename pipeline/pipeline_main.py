from pipeline.optimization_workflow.optimization_manager import OptManager

max_iter = 8
dir_name = 'sindy-burg'
start_iter = 0
refine_point = 100

debug = True # True False
print_exc = True
exit_code = True


if __name__ == '__main__':
    opt_manager = OptManager(max_iter, start_iter, refine_point, dir_name, debug, print_exc, exit_code,
                             resample_shape=(20, 20), n_candidates=4)
    opt_manager.explore_solutions()
    pruned_track, by_project_track = opt_manager.call_pruner()
    print()



