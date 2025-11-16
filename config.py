config = {

    "Short_to_Long_Label" : {
        'ST':'straight',
        'RT':'right_turn',
        'LT':'left_turn',
        'UT':'U_turn',
        'LLC':'left_lane_change',
        'RLC':'right_lane_change',
        'RA':'roundabout'
    },

    "Long_to_Short_Label" : {
        'straight':'ST',
        'right_turn':'RT',
        'left_turn':'LT',
        'U_turn':'UT',
        'left_lane_change':'LLC',
        'right_lane_change':'RLC',
        'roundabout':'RA'
    },

    "class_to_label":{
        1: "straight",
        2: "right_turn",
        3: "left_turn",
        4: "U_turn",
        5: "left_lane_change",
        6: "right_lane_change",
        7: "roundabout",
        },

    'label_to_class' :{
        'straight': 1,
        'right_turn': 2,
        'left_turn': 3,
        'U_turn': 4,
        'left_lane_change': 5,
        'right_lane_change': 6,
        'roundabout': 7
    },

    'data_columns':[
        'time (sec)', 'PositionX (m)',
        'PositionY (m)','PositionZ (m)', 'VelocityX(EntityCoord) (km/h)',
        'VelocityY(EntityCoord) (km/h)', 'VelocityZ(EntityCoord) (km/h)',
        'AccelerationX(EntityCoord) (m/s2)',
        'AccelerationY(EntityCoord) (m/s2)',
        'AccelerationZ(EntityCoord) (m/s2)',
        'RotationZ (deg)', 'FrontWheelAngle (deg)'],

    "SAMPLING_RATE": 50,  # Hz 1/0.02

    "UNCLE_DIR_NAME": "simulation_TOTAL_250626"
}
