#!/bin/bash

# for np in 4 2 1 ; do
  for ksptype in tfqmr gmres bcgs ; do
    for pctype in bjacobi asm ; do
      for subpctype in jacobi sor ilu ; do
        if [ $subpctype == ilu ]; then
          for level in 0 1 2 ; do
            # echo "========================================================================================================================"
            # echo "==                                                                                                                    =="
            # echo "==                                                                                                                    =="
            # echo "==       New Run                                                                                                      =="
            # echo "==                                                                                                                    =="
            # echo "==                                                                                                                    =="
            # echo "========================================================================================================================"
            # mpiexec -n $np ../../bin/TDSE  -prop_ksp_type $ksptype -prop_pc_type $pctype -prop_sub_ksp_type preonly -prop_sub_pc_type $subpctype -prop_sub_pc_ilu_levels $level -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist -log_view -options_left
            echo $ksptype $pctype $subpctype $level
          done
        else
          # echo "========================================================================================================================"
          # echo "==                                                                                                                    =="
          # echo "==                                                                                                                    =="
          # echo "==       New Run                                                                                                      =="
          # echo "==                                                                                                                    =="
          # echo "==                                                                                                                    =="
          # echo "========================================================================================================================"
          # mpiexec -n $np ../../bin/TDSE  -prop_ksp_type $ksptype -prop_pc_type $pctype -prop_sub_ksp_type preonly -prop_sub_pc_type $subpctype -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist -log_view -options_left
          echo $ksptype $pctype $subpctype
        fi
      done
    done
  done
# done
# mpiexec -n 4 ../../bin/TDSE  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist -log_view -options_left

