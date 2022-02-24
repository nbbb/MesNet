class MSVDSplitConfig:
    model = "MSVD_InceptionV4"

    video_fpath = "/data/ntz/ntz_code2/Data/MSVD/Feats/CNN_I3D.hdf5"
    caption_fpath = "data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "/data/ntz/ntz_code2/Data/MSVD/phase_feats/CNN_I3D_train.hdf5"
    val_video_fpath = "/data/ntz/ntz_code2/Data/MSVD/phase_feats/CNN_I3D_val.hdf5"
    test_video_fpath = "/data/ntz/ntz_code2/Data/MSVD/phase_feats/CNN_I3D_test.hdf5"

    train_metadata_fpath = "data/MSVD/metadata/train.csv"
    val_metadata_fpath = "data/MSVD/metadata/val.csv"
    test_metadata_fpath = "data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig:
    model = "MSR-VTT_InceptionV4"

    video_fpath = "/data2/ntz/ntz_code2/Data/MSRVTT/Feats/MSR-VTT_I3D.hdf5"
    train_val_caption_fpath = "../data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "../data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_I3D_r1_train.hdf5"
    val_video_fpath = "/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_I3D_r1_val.hdf5"
    test_video_fpath = "/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_I3D_r1_test.hdf5"

    train_metadata_fpath = "/data/ntz/ntz_code2/Data/MSRVTT/metadata/train_r3.json"
    val_metadata_fpath = "/data/ntz/ntz_code2/Data/MSRVTT/metadata/val_r3.json"
    test_metadata_fpath = "/data/ntz/ntz_code2/Data/MSRVTT/metadata/test_r3.json"

