#pragma once

typedef float real;

int tri_tri_overlap_test_3d(real p1[3], real q1[3], real r1[3],
    real p2[3], real q2[3], real r2[3]);


int coplanar_tri_tri3d(real  p1[3], real  q1[3], real  r1[3],
    real  p2[3], real  q2[3], real  r2[3],
    real  N1[3], real  N2[3]);


int tri_tri_overlap_test_2d(real p1[2], real q1[2], real r1[2],
    real p2[2], real q2[2], real r2[2]);


int tri_tri_intersection_test_3d(real p1[3], real q1[3], real r1[3],
    real p2[3], real q2[3], real r2[3],
    int* coplanar,
    real source[3], real target[3]);

