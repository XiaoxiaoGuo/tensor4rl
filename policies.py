
def encode(taxirow, taxicol, passloc, destidx):
    # (5) 5, 5, 4
    i = taxirow
    i *= 5
    i += taxicol
    i *= 5
    i += passloc
    i *= 4
    i += destidx
    return i


def decode_location(i):
    if i == 0:
        return 0, 0
    elif i == 1:
        return 0, 4
    elif i == 2:
        return 4, 0
    elif i == 3:
        return 4, 3


def decode(i):
    destidx = i % 4
    i = i // 4
    passloc = i % 5
    i = i // 5
    taxicol = i % 5
    i = i // 5
    taxirow = i
    assert 0 <= i < 5
    return taxirow, taxicol, passloc, destidx


# South (0)
# North (1)
# East (2)
# West (3)
# Pickup (4)
# Drop-off (5)
def optimal_policy(i):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    taxirow, taxicol, passloc, destidx = decode(i)
    if passloc < 4: # To pick up passenger
        passrow, passcol = decode_location(passloc)
        if passcol == taxicol:
            if taxirow == passrow:
                return PICKUP
            elif taxirow > passrow:
                return NORTH
            return SOUTH
        else:
            if taxirow == 1 or taxirow == 2:
                if passcol < taxicol:
                    return WEST
                return EAST
            else:
                if taxirow < 1:
                    return SOUTH
                return NORTH
    else: # To goal location
        destrow, destcol = decode_location(destidx)
        if destcol == taxicol:
            if taxirow == destrow:
                return DROPOFF
            elif taxirow > destrow:
                return NORTH
            return SOUTH
        else:
            if taxirow == 1 or taxirow == 2:
                if destcol < taxicol:
                    return WEST
                return EAST
            else:
                if taxirow < 1:
                    return SOUTH
                return NORTH
    return -1

def get_next_state(state, action):
    taxirow, taxicol, passloc, destidx = decode(state)
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    if action == PICKUP:
        if passloc < 4:
            passrow, passcol = decode_location(passloc)
            if passrow == taxirow and passcol == taxicol:
                return encode(taxirow, taxicol, 4, destidx)
        return state
    elif action == DROPOFF:
        if passloc == 4:
            destrow, destcol = decode_location(destidx)
            if destrow == taxirow and destcol == taxicol:
                return 500
        return state
    elif action == NORTH:
        if taxirow > 0:
            return encode(taxirow-1, taxicol, passloc, destidx)
        return state
    elif action == SOUTH:
        if taxirow < 4:
            return encode(taxirow+1, taxicol, passloc, destidx)
        return state
    elif action == WEST:
        if taxicol == 1:
            if taxirow < 3:
                return encode(taxirow, taxicol-1, passloc, destidx)
        elif taxicol == 2:
            if taxirow > 0:
                return encode(taxirow, taxicol-1, passloc, destidx)
        elif taxicol == 3:
            if taxirow < 3:
                return encode(taxirow, taxicol-1, passloc, destidx)
        elif taxicol == 4:
            return encode(taxirow, taxicol - 1, passloc, destidx)
        return state
    elif action == EAST:
        if taxicol == 0:
            if taxirow < 3:
                return encode(taxirow, taxicol + 1, passloc, destidx)
        elif taxicol == 1:
            if taxirow > 0:
                return encode(taxirow, taxicol + 1, passloc, destidx)
        elif taxicol == 2:
            if taxirow < 3:
                return encode(taxirow, taxicol + 1, passloc, destidx)
        elif taxicol == 3:
            return encode(taxirow, taxicol + 1, passloc, destidx)
        return state
    return -1


# eta: state coverage rate
# eps: action randomness
def generate_fixed_data_set(eta, eps):
    # apply state removal
    cur_state = [i for i in range(500)]
    import random
    import math
    num_missing_state = math.floor(500 * eta)
    missing_state = []
    for i in range(num_missing_state):
        candidate = random.choice(cur_state)
        missing_state.append(candidate)
        cur_state.remove(candidate)

    next_state = []
    opt_action = []
    # apply action intervention
    random_count = 0
    for i in range(len(cur_state)):
        if random.random() <= eps:
            random_count += 1
            act = random.randrange(0, 4)
        else:
            act = optimal_policy(cur_state[i])
        opt_action.append(act)
        next_state.append(get_next_state(cur_state[i], act))
    print('act random rate:', random_count / len(cur_state), '\t #state:', len(cur_state))
    print('missing states:', missing_state)
    return cur_state, opt_action, next_state

#
# cur, act, nex = generate_fixed_data_set(0.5,0.1)
# print('cur:', len(cur))
# print('act:', len(act))
# print('nex:', len(nex))
# print(decode(165))
