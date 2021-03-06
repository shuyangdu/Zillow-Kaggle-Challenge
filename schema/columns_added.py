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
    # 'area_basic',  # perfectly correlates with area_total_finished_calc
    # 'area_live_finished',  # perfectly correlates with area_total_finished_calc
    # 'area_liveperi_finished',  # perfectly correlates with area_total_finished_calc
    # 'area_total_finished',  # perfectly correlates with area_total_finished_calc
    # 'area_live_entry_finished',  # perfectly correlates with area_firstfloor_finished
    'num_unit',
    'num_story',
    'num_room',
    'num_bathroom',
    'num_bedroom',
    # 'num_bathroom_calc',  # perfectly correlates with num_bathroom
    # 'num_bathroom_full',  # perfectly correlates with num_bathroom
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
    # ------ features added ------
    'ratio_value_land_vs_building',
    'ratio_value_tax_property_vs_total',
    'ratio_area_living_vs_total',
    'avg_city_area_lot',
    'avg_city_area_total_finished_calc',
    'avg_city_value_building',
    'avg_city_value_tax_property',
    'avg_city_value_total',
    'avg_zip_area_lot',
    'avg_zip_area_total_finished_calc',
    'avg_zip_value_building',
    'avg_zip_value_tax_property',
    'avg_zip_value_total',
    'deviation_city_area_lot',
    'deviation_city_area_total_finished_calc',
    'deviation_city_value_building',
    'deviation_city_value_tax_property',
    'deviation_city_value_total',
    'deviation_zip_area_lot',
    'deviation_zip_area_total_finished_calc',
    'deviation_zip_value_building',
    'deviation_zip_value_tax_property',
    'deviation_zip_value_total',
    'count_properties_zip',
    'count_properties_city',
    'count_properties_neighbor',
    'latitude_cos',
    'longitude_cos',
    'latitude_times_longitude',
    'area_room_avg',
    'num_extra_room',
    'area_extra',
]

LOG_COLS = [
    'area_patio',
    'area_shed',
    'area_pool',
    'area_lot',
    'area_garage',
    'area_firstfloor_finished',
    'area_total_finished_calc',
    # 'area_basic',
    # 'area_live_finished',
    # 'area_liveperi_finished',
    # 'area_total_finished',
    # 'area_live_entry_finished',
    'num_unit',
    'num_room',
    'num_bathroom',
    'num_bedroom',
    # 'num_bathroom_calc',
    # 'num_bathroom_full',
    'num_bathroom_quarter',
    'num_fireplace',
    'num_garage',
    'value_total',
    'value_building',
    'value_land',
    'value_tax_property',
    # 'tax_year',  # only one value, deleted
    'tax_delinquency_year',
    # ------ features added ------
    'avg_city_area_lot',
    'avg_city_area_total_finished_calc',
    'avg_city_value_building',
    'avg_city_value_tax_property',
    'avg_city_value_total',
    'avg_zip_area_lot',
    'avg_zip_area_total_finished_calc',
    'avg_zip_value_building',
    'avg_zip_value_tax_property',
    'avg_zip_value_total',
    'ratio_value_land_vs_building',
    'ratio_value_tax_property_vs_total',
    'ratio_area_living_vs_total',
    'area_room_avg',
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
    # 'flag_fireplace',  # use num_firepalce instead
    'flag_hot_tub',
    'flag_spa',
    'flag_no_tub_or_spa',
    # 'flag_tub',  included by flag_hot_tub and flag_spa
    'framing',
    'material',
    'deck',
    'story',
    'heating',
    'aircon',
    'architectural_style',
    'fips',
    # ------ feature added ------
    'transaction_month',  # added feature based on transaction date
    # 'flag_multiple_sales',
    # 'region_county_filled',
    # 'region_city_filled',
    # 'region_zip_filled',
    # 'region_neighbor_filled',
    # 'flag_nan_region_zip',
    # 'flag_nan_region_city',
    # 'flag_nan_region_neighbor',
    # 'flag_nan_region_county',
]

LABEL_COL = 'logerror'
