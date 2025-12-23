#ifndef _SK_MOBILE_ROBOT_COMMON_H_
#define _SK_MOBILE_ROBOT_COMMON_H_

// pre-defined operators
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define TOGGLE(x) (x = (x+1)%2)
#define RAD2DEG (180.0/M_PI)
#define DEG2RAD (M_PI/180.0)
#define SQRT2 (sqrt(2.0))

#define SK_OFFSET_MAGNETIC_TRUE_NORTH_OFFSET    (7.0*DEG2RAD)

// Robot State
#define SK_MOBILE_ROBOT_IDLE                    (0)
#define SK_MOBILE_ROBOT_STOP                    (1)
#define SK_MOBILE_ROBOT_GOTO_POSE               (5)
#define SK_MOBILE_ROBOT_GOTO_POSE_FORCE         (6)
#define SK_MOBILE_ROBOT_POSE_PATH               (7)
#define SK_MOBILE_ROBOT_POSE_PATH_FORCE         (8)


#define SK_MOBILE_ROBOT_ACTION_RUNNING          (9)

#define SK_MOBILE_ROBOT_DASH_INIT               (10)
#define SK_MOBILE_ROBOT_DASH                    (11)
#define SK_MOBILE_ROBOT_DASH_FINISH             (12)
#define SK_MOBILE_ROBOT_TURN_IN_POSITION        (13)

#define SK_MOBILE_ROBOT_LINE_FOLLOW_INIT        (20)
#define SK_MOBILE_ROBOT_LINE_FOLLOW             (21)
#define SK_MOBILE_ROBOT_LINE_FOLLOW_FINISH      (22)

#define SK_MOBILE_ROBOT_DRIVE_GPS               (30)

// Dash Type
#define SK_MOBILE_ROBOT_DASH_NONE                       (0)
#define SK_MOBILE_ROBOT_DASH_CANCEL                     (1)
#define SK_MOBILE_ROBOT_DASH_PERIOD                     (2)
#define SK_MOBILE_ROBOT_DASH_SPEED                      (0.3)
#define SK_MOBILE_ROBOT_DASH_DURATION                   (1.0)

#define SK_MOBILE_ROBOT_MANUAL                  (10)
#define SK_MOBILE_ROBOT_MANUAL_IGNORE_SENSOR    (12)
#define SK_MOBILE_ROBOT_MANUAL_MEASURE_SPEED    (14)

// LineFollow Type
#define SK_MOBILE_ROBOT_WALL_FOLLOW_SPEED               (0.3)
#define SK_MOBILE_ROBOT_WALL_FOLLOW_SIDE_DIST           (0.5)
#define SK_MOBILE_ROBOT_WALL_FOLLOW_STOP_DIST           (0.5)

#define SK_MOBILE_ROBOT_MANUAL                  (10)
#define SK_MOBILE_ROBOT_MANUAL_IGNORE_SENSOR    (12)
#define SK_MOBILE_ROBOT_MANUAL_MEASURE_SPEED    (14)


/*
#define SK_MOBILE_ROBOT_AUTO                    (20)
#define SK_MOBILE_ROBOT_FOLLOW_LEFT             (21)
#define SK_MOBILE_ROBOT_FOLLOW_RIGHT            (22)
#define SK_MOBILE_ROBOT_FOLLOW_CENTER           (23)
#define SK_MOBILE_ROBOT_AUTO_EVAL               (24)

#define SK_MOBILE_ROBOT_SMART_FARM_ENTER (32)
#define SK_MOBILE_ROBOT_SMART_FARM_MOVEBACK (33)
#define SK_MOBILE_ROBOT_SMART_FARM_LEAVE (34)
#define SK_MOBILE_ROBOT_SMART_FARM_RESET (35)
#define SK_MOBILE_ROBOT_SMART_FARM_FORCE_ON_RAIL (36)
#define SK_MOBILE_ROBOT_SMART_FARM_ON_RAIL      (37)
#define SK_MOBILE_ROBOT_SMART_FARM_FORCE_OFF_RAIL (38)
#define SK_MOBILE_ROBOT_SMART_FARM_CHARGE       (39)
#define SK_MOBILE_ROBOT_SMART_FARM_BOOST       (40)

#define SK_MOBILE_ROBOT_CAM_CALIBRATION (50)
*/
// Errors over 100
#define SK_MOBILE_ROBOT_ERROR (100)
/*
// Robot Type
#define SK_MOBILE_ROBOT_KRW_MOBILE (10)
#define SK_MOBILE_ROBOT_KRW_DISINF (11)
#define SK_MOBILE_ROBOT_ERP_MINI (20)
#define SK_MOBILE_ROBOT_SMART_FARM (30)
#define SK_MOBILE_ROBOT_JETBOT (40)
#define SK_MOBILE_ROBOT_TUT_0 (50)              // Turtle bot Proto type
#define SK_MOBILE_ROBOT_TUT_2 (52)              // Turtle bot 2 wheel differential drive version
#define SK_MOBILE_ROBOT_TUT_3 (53)              // Turtle bot 3 omni wheel version

// Joy Type
//#define SK_MOBILE_JOY_NONE (0)
//#define SK_MOBILE_JOY_F710 (10)
//#define SK_MOBILE_JOY_EX3D (20)

// Index for Disinfection Robot
#define SK_MOBILE_DISINFECTION_UV_STATE (0)
#define SK_MOBILE_DISINFECTION_UV_BODY (1)
#define SK_MOBILE_DISINFECTION_UV_BOTTOM (2)
#define SK_MOBILE_DISINFECTION_SMOKE_STATE (3)
#define SK_MOBILE_DISINFECTION_HEAD_STATE (4)
#define SK_MOBILE_DISINFECTION_PAN (5)
#define SK_MOBILE_DISINFECTION_TILT (6)
#define SK_MOBILE_DISINFECTION_LENGTH (7)
*/
/*
#define SK_MOBILE_DISINFECTION_UV_MAUNAL (0)
#define SK_MOBILE_DISINFECTION_UV_AUTO (1)
#define SK_MOBILE_DISINFECTION_UV_MAX (2)

#define SK_MOBILE_DISINFECTION_HEAD_MAUNAL (0)
#define SK_MOBILE_DISINFECTION_HEAD_AUTO (1)
#define SK_MOBILE_DISINFECTION_HEAD_MAX (2)

#define SK_MOBILE_DISINFECTION_SMOKE_OFF (0)
#define SK_MOBILE_DISINFECTION_SMOKE_ON (1)
#define SK_MOBILE_DISINFECTION_SMOKE_AUTO (2)
#define SK_MOBILE_DISINFECTION_SMOKE_AIM (3)
#define SK_MOBILE_DISINFECTION_SMOKE_MANUAL (4)
#define SK_MOBILE_DISINFECTION_SMOKE_MAX (2)
*/

struct sPoint2D
{
    double x;
    double y;
};

struct sLine
{
    double dist;
    double angle;
    bool valid;
};

#endif // _SK_MOBILE_ROBOT_COMMON_H_
