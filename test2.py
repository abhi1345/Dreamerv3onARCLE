import arcle
from onehotsel import OneHotSel

env = arcle.envs.arcenv.RawARCEnv()
env = OneHotSel()

print(env.action_space)