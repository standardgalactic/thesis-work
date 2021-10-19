# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 12:41:51 2021

@author: Mark Zaidi
"""
import cProfile
import pstats
import snakeviz
import Data_Visualization_v2
with cProfile.Profile() as pr:
    Data_Visualization_v2.main()
stats=pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
#stats.print_stats()
stats.dump_stats(filename='profiling_results.prof')
!snakeviz profiling_results.prof