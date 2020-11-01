conf = {
        "WORK_PATH": "./work/",
        "CUDA_VISIBLE_DEVICES": "0",

        "data": {
            'dataset_path': "YOUR DATASET PATH",
            'resolution': '64',
            'dataset': 'OUMVLP',
            'pid_num': 5153,
            'pid_shuffle': False,
        },
        "model": {
            'lr': 1e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (32, 8),
            'restore_iter': 0,
            'total_iter': 250000,
            'margin': 0.2,
            'num_workers': 0,
            'frame_num': 30,

            'model_name': 'GaitGL_OUMVLP'  



        }
    }
