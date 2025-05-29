
config = {
    "class_to_label":{
        1: "Straight",
        2: "Left",
        3: "Right",
        4: "U-turn",
        5: "Roundabout",
        6: "Lane Change - Left",
        7: "Lane Change - Right",
        8: "Merge - Left",
        9: "Merge - Right",
        10: "Throwing"
        },

    'label_to_class' :{
        'Straight': 1,
        'Left': 2,
        'Right': 3,
        'U-turn': 4,
        'Roundabout': 5,
        'Lane Change - Left': 6,
        'Lane Change - Right': 7,
        'Merge - Left': 8,
        'Merge - Right': 9,
        'Throwing': 10
    },

    'data_columns':[
        'time (sec)', 'PositionX (m)', 
        'PositionY (m)','PositionZ (m)', 'VelocityX(EntityCoord) (km/h)',
        'VelocityY(EntityCoord) (km/h)', 'VelocityZ(EntityCoord) (km/h)',
        'AccelerationX(EntityCoord) (m/s2)',
        'AccelerationY(EntityCoord) (m/s2)',
        'AccelerationZ(EntityCoord) (m/s2)', 
        'RotationZ (deg)', 'FrontWheelAngle (deg)'],
}


