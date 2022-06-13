NUMBER_OF_CELLS = 3
MOVE_SUCCESS = 0.8
SENSOR_PROB = {
    # IS_WALL: ON, OFF
    False: [0.1, 0.9],
    True: [0.6, 0.4]
}



def calc(pos: int, events: list, is_initial=True) -> float:
    if not events:
        return 1.0

    event_type: int = 0 if events[0] == 'on' else 1

    if is_initial:
        return calc(pos, events[1:], False) * SENSOR_PROB[pos == 2][event_type]
    else:
        if pos == 0:
            return calc(pos, events[1:], False) * SENSOR_PROB[False][event_type] * (1.0 - MOVE_SUCCESS)
        elif pos == 1:
            return calc(pos-1, events[1:], False) * SENSOR_PROB[False][event_type] * MOVE_SUCCESS\
                +  calc(pos, events[1:], False) * SENSOR_PROB[False][event_type] * (1.0 - MOVE_SUCCESS)
        elif pos == 2:
            return calc(pos-1, events[1:], False) * SENSOR_PROB[False][event_type] * MOVE_SUCCESS\
                +  calc(pos, events[1:], False) * SENSOR_PROB[True][event_type]
        else:
            assert 0

def normalize(input_list: list) -> list:
    mult = 1 / sum(input_list)
    return [e * mult for e in input_list]

def main(args=None):
    events = input()
    events = events.lower().split()
    events.reverse()
    # print("EVENTS (Reversed):", events)
    ans = []
    
    for i in range(NUMBER_OF_CELLS):
        ans.append(calc(i, events))
    # print(ans)
    ans = normalize(ans)

    print("LEFT :", ans[0])
    print("MID  :", ans[1])
    print("RIGHT:", ans[2])

if __name__ == '__main__':
    main()