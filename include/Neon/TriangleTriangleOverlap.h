#pragma once

typedef float _real;

int tri_tri_overlap_test_3d(_real p1[3], _real q1[3], _real r1[3],
    _real p2[3], _real q2[3], _real r2[3]);


int coplanar_tri_tri3d(_real  p1[3], _real  q1[3], _real  r1[3],
    _real  p2[3], _real  q2[3], _real  r2[3],
    _real  N1[3], _real  N2[3]);


int tri_tri_overlap_test_2d(_real p1[2], _real q1[2], _real r1[2],
    _real p2[2], _real q2[2], _real r2[2]);


int tri_tri_intersection_test_3d(_real p1[3], _real q1[3], _real r1[3],
    _real p2[3], _real q2[3], _real r2[3],
    int* coplanar,
    _real source[3], _real target[3]);

