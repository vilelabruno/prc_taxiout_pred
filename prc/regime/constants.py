APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
WINDOWS = (15, 30, 60, 120)
TRAIN_MONTHS = (1, 7)
Y_MAX = 50000.0
RAW_COLUMNS = [KEY, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt',
               'RUNWAY_mvt', 'STAND_mvt', 'SCHED_TIME_UTC_mvt']
BASE_FEATURES = ['aptc', 'mvt_aobt', 'mvt_sched', 'mvt_eobt', 'hr']
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.7,
              lambda_l2=10, verbose=-1)
SEEDS = (11, 23, 37)
ROUNDS = 300
DOSE = 0.75
ZERO_AIRPORTS = ('EDDM', 'EHAM', 'LSZH')
LAMBDA = {1: 0.89, 7: 0.89, 2: 0.6, 6: 0.6}
FLOOR = 60.0
