from preprocess_funcs import extract_edf_data

base_path = '../DATA/MASS/'
edf_files = [base_path + '01-02-%s.edf' % str((i + 1)).zfill(4) for i in range(20)]
# base_path = '../DATA/SEEG/'
# edf_files = [base_path + 'patient_%s.edf' % str((i + 1)).zfill(4) for i in range(5)]
annotations = [[base_path + 'annotations/SpindleE1/01-02-0001_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0001_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0002_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0002_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0003_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0003_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0004_SpindleE1.edf', ],
               [base_path + 'annotations/SpindleE1/01-02-0005_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0005_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0006_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0006_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0007_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0007_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0008_SpindleE1.edf', ],
               [base_path + 'annotations/SpindleE1/01-02-0009_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0009_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0010_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0010_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0011_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0011_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0012_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0012_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0013_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0013_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0014_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0014_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0015_SpindleE1.edf', ],
               [base_path + 'annotations/SpindleE1/01-02-0016_SpindleE1.edf', ],
               [base_path + 'annotations/SpindleE1/01-02-0017_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0017_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0018_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0018_SpindleE2.edf'],
               [base_path + 'annotations/SpindleE1/01-02-0019_SpindleE1.edf',
                base_path + 'annotations/SpindleE2/01-02-0019_SpindleE2.edf']]

params = {'window_duration': 4.0,   # duration of segmented signal in seconds
          'overlap': 0.5,           # decimal percent overlap of segmented signal
          'bandpass': (8, 17),      # (low frequency cutoff, high frequency cutoff)
          'nfft': 1024,             # number of points for FFT (zero padded)
          'spec_window': 0.125,     # duration of spectrogram window in seconds
          'spec_overlap': 0.0625,   # duration of step between spectrogram segments
          'reference': ('A1', 'A2'),
          'f_crop': (8, 17)}

name_func = lambda s: s.split(' ')[1].split('-')[0]

extract_edf_data(edf_files, params, base_path=base_path, annotations=annotations, name_func=name_func)
