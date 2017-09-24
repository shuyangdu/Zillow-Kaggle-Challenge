NUMERICAL_COLS = [
    'build_year',
    'area_basement',
    'area_patio',
    'area_shed',
    'area_pool',
    'area_lot',
    'area_garage',
    'area_firstfloor_finished',
    'area_total_finished_calc',
    'area_basic',  # perfectly correlates with area_total_finished_calc
    'area_live_finished',  # perfectly correlates with area_total_finished_calc
    'area_liveperi_finished',  # perfectly correlates with area_total_finished_calc
    'area_total_finished',  # perfectly correlates with area_total_finished_calc
    'area_live_entry_finished',  # perfectly correlates with area_firstfloor_finished
    'num_unit',
    'num_story',
    'num_room',
    'num_bathroom',
    'num_bedroom',
    'num_bathroom_calc',  # perfectly correlates with num_bathroom
    'num_bathroom_full',  # perfectly correlates with num_bathroom
    'num_bathroom_quarter',
    'num_fireplace',
    'num_pool',
    'num_garage',
    'value_total',
    'value_building',
    'value_land',
    'value_tax_property',
    # 'tax_year',  # only one value, deleted
    'tax_delinquency_year',
    'quality',
    'latitude',
    'longitude',
]

LOG_COLS = [
    'area_patio',
    'area_shed',
    'area_pool',
    'area_lot',
    'area_garage',
    'area_firstfloor_finished',
    'area_total_finished_calc',
    'area_basic',
    'area_live_finished',
    'area_liveperi_finished',
    'area_total_finished',
    'area_live_entry_finished',
    'num_unit',
    'num_room',
    'num_bathroom',
    'num_bedroom',
    'num_bathroom_calc',
    'num_bathroom_full',
    'num_bathroom_quarter',
    'num_fireplace',
    'num_garage',
    'value_total',
    'value_building',
    'value_land',
    'value_tax_property',
    # 'tax_year',  # only one value, deleted
    'tax_delinquency_year',
]

CATEGORICAL_COLS = [
    'region_county',
    'region_city',
    'region_zip',
    'region_neighbor',
    'tax_flag_delinquency',
    'zoning_property',
    'zoning_landuse',
    'zoning_landuse_county',
    'flag_fireplace',  # use num_firepalce instead
    'flag_hot_tub',
    'flag_spa',
    'flag_no_tub_or_spa',
    # 'flag_tub',  # included by flag_hot_tub and flag_spa
    'framing',
    'material',
    'deck',
    'story',
    'heating',
    'aircon',
    'architectural_style',
    'fips',
    'transaction_month',  # added feature based on transaction date
]

LABEL_COL = 'logerror'
