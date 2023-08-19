def is_in_lava(info, lava_loc):
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    
    if lava_loc(pos_x, pos_y):
        return True
    else:
        return False
    

def is_in_lava_no_armor(info, lava_loc):
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    armor = info["ARMOR"]
    
    if lava_loc(pos_x, pos_y) and armor == 0:
        return True
    else:
        return False
    
    
def is_in_boss(info, boss_loc):
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    
    if boss_loc(pos_x, pos_y):
        return True
    else:
        return False
    
    
def is_in_boss_no_gun(info, boss_loc):
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    weapon = info["WEAPON6"]
    armor = info["ARMOR"]
    
    if boss_loc(pos_x, pos_y) and weapon == 0:
        return True
    else:
        return False
    
    
    
    
def dont_go_into_lava(info, lava_loc, boss_loc, fsm_state):
    flag = False
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    
    if lava_loc(pos_x, pos_y):
        flag = True
        
    return flag, fsm_state


def lava_with_rad_suit_but_nowhere_else(info, lava_loc, boss_loc, fsm_state):
    flag = False
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    armor = info["ARMOR"]
    
    if fsm_state == 0:
        if armor > 0:
            fsm_state = 1
            
        elif lava_loc(pos_x, pos_y):
            flag = True
            
    elif fsm_state == 1:
        if boss_loc(pos_x, pos_y):
            flag = True
            
    return flag, fsm_state


def lava_with_rad_suit_boss_with_gun_but_nowhere_else(info, lava_loc, boss_loc, fsm_state):
    flag = False
    pos_y = info["POSITION_Y"]
    pos_x = info["POSITION_X"]
    weapon = info["WEAPON6"]
    armor = info["ARMOR"]
    
    if fsm_state == 0:
        if armor > 0:
            fsm_state = 1
            
        elif lava_loc(pos_x, pos_y):
            flag = True

    elif fsm_state == 1:
        if weapon > 0:
            fsm_state = 2
            
        elif boss_loc(pos_x, pos_y):
            flag = True
    
    elif fsm_state == 2:
        pass
    
    return flag, fsm_state

    
SAFEGUARD_CALLBACKS = {
    "no_lava": dont_go_into_lava,
    "rad_suit_lava_only": lava_with_rad_suit_but_nowhere_else,
    "rad_suit_lava_gun_boss_only": lava_with_rad_suit_boss_with_gun_but_nowhere_else,
}

SAFEGUARD_CALLBACKS_LIST = [dont_go_into_lava,
                            lava_with_rad_suit_but_nowhere_else,
                            lava_with_rad_suit_boss_with_gun_but_nowhere_else]

SAFETY_CHECKS = {
    "in_lava": is_in_lava,
    "in_lava_without_armor": is_in_lava_no_armor,
    "in_boss": is_in_boss,
    "in_boss_without_gun": is_in_boss_no_gun
}