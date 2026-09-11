#include "cshape.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
// **************************************************

// 3D Shape calculation

// **************************************************


typedef struct {
  int z2;
  int y2;
  int x2;
} Vertex3I;

typedef struct {
  int y2;
  int x2;
} Vertex2I;

typedef struct {
  size_t start;
  size_t end;
  int axis;
  int left;
  int right;
  int min_c[3];
  int max_c[3];
} KDNode3;

typedef struct {
  size_t start;
  size_t end;
  int axis;
  int left;
  int right;
  int min_c[2];
  int max_c[2];
} KDNode2;

#define KD3_LEAF_SIZE 32
#define KD2_LEAF_SIZE 32

void calculate_meshDiameter(const Vertex3I *vertices, size_t count, const double *spacing, double *diameters);
double calculate_meshDiameter2D(const Vertex2I *points, size_t count, const double *spacing);

static int compare_vertex3_zyx(const void *a, const void *b)
{
  const Vertex3I *va = (const Vertex3I *)a;
  const Vertex3I *vb = (const Vertex3I *)b;
  if (va->z2 != vb->z2) return (va->z2 < vb->z2) ? -1 : 1;
  if (va->y2 != vb->y2) return (va->y2 < vb->y2) ? -1 : 1;
  if (va->x2 != vb->x2) return (va->x2 < vb->x2) ? -1 : 1;
  return 0;
}

static int compare_vertex3_z_yx(const void *a, const void *b)
{
  return compare_vertex3_zyx(a, b);
}

static int compare_vertex3_y_zx(const void *a, const void *b)
{
  const Vertex3I *va = (const Vertex3I *)a;
  const Vertex3I *vb = (const Vertex3I *)b;
  if (va->y2 != vb->y2) return (va->y2 < vb->y2) ? -1 : 1;
  if (va->z2 != vb->z2) return (va->z2 < vb->z2) ? -1 : 1;
  if (va->x2 != vb->x2) return (va->x2 < vb->x2) ? -1 : 1;
  return 0;
}

static int compare_vertex3_x_zy(const void *a, const void *b)
{
  const Vertex3I *va = (const Vertex3I *)a;
  const Vertex3I *vb = (const Vertex3I *)b;
  if (va->x2 != vb->x2) return (va->x2 < vb->x2) ? -1 : 1;
  if (va->z2 != vb->z2) return (va->z2 < vb->z2) ? -1 : 1;
  if (va->y2 != vb->y2) return (va->y2 < vb->y2) ? -1 : 1;
  return 0;
}

static int compare_vertex2_yx(const void *a, const void *b)
{
  const Vertex2I *va = (const Vertex2I *)a;
  const Vertex2I *vb = (const Vertex2I *)b;
  if (va->y2 != vb->y2) return (va->y2 < vb->y2) ? -1 : 1;
  if (va->x2 != vb->x2) return (va->x2 < vb->x2) ? -1 : 1;
  return 0;
}

static int kd3_sort_axis = 0;
static int kd2_sort_axis = 0;

static int compare_vertex3_axis(const void *a, const void *b)
{
  const Vertex3I *va = (const Vertex3I *)a;
  const Vertex3I *vb = (const Vertex3I *)b;
  int ca = (kd3_sort_axis == 0) ? va->z2 : (kd3_sort_axis == 1 ? va->y2 : va->x2);
  int cb = (kd3_sort_axis == 0) ? vb->z2 : (kd3_sort_axis == 1 ? vb->y2 : vb->x2);
  if (ca != cb) return (ca < cb) ? -1 : 1;
  return compare_vertex3_zyx(a, b);
}

static int compare_vertex2_axis(const void *a, const void *b)
{
  const Vertex2I *va = (const Vertex2I *)a;
  const Vertex2I *vb = (const Vertex2I *)b;
  int ca = (kd2_sort_axis == 0) ? va->y2 : va->x2;
  int cb = (kd2_sort_axis == 0) ? vb->y2 : vb->x2;
  if (ca != cb) return (ca < cb) ? -1 : 1;
  return compare_vertex2_yx(a, b);
}

static size_t dedup_vertex3(Vertex3I *points, size_t count)
{
  size_t i, write_idx;
  if (count == 0)
    return 0;
  qsort(points, count, sizeof(Vertex3I), compare_vertex3_zyx);
  write_idx = 1;
  for (i = 1; i < count; i++)
  {
    if (compare_vertex3_zyx(&points[i], &points[write_idx - 1]) != 0)
      points[write_idx++] = points[i];
  }
  return write_idx;
}

static size_t dedup_vertex2(Vertex2I *points, size_t count)
{
  size_t i, write_idx;
  if (count == 0)
    return 0;
  qsort(points, count, sizeof(Vertex2I), compare_vertex2_yx);
  write_idx = 1;
  for (i = 1; i < count; i++)
  {
    if (compare_vertex2_yx(&points[i], &points[write_idx - 1]) != 0)
      points[write_idx++] = points[i];
  }
  return write_idx;
}

static void kd3_compute_bbox(const Vertex3I *points, size_t start, size_t end, int *min_c, int *max_c)
{
  size_t i;
  min_c[0] = max_c[0] = points[start].z2;
  min_c[1] = max_c[1] = points[start].y2;
  min_c[2] = max_c[2] = points[start].x2;
  for (i = start + 1; i < end; i++)
  {
    if (points[i].z2 < min_c[0]) min_c[0] = points[i].z2;
    if (points[i].z2 > max_c[0]) max_c[0] = points[i].z2;
    if (points[i].y2 < min_c[1]) min_c[1] = points[i].y2;
    if (points[i].y2 > max_c[1]) max_c[1] = points[i].y2;
    if (points[i].x2 < min_c[2]) min_c[2] = points[i].x2;
    if (points[i].x2 > max_c[2]) max_c[2] = points[i].x2;
  }
}

static void kd2_compute_bbox(const Vertex2I *points, size_t start, size_t end, int *min_c, int *max_c)
{
  size_t i;
  min_c[0] = max_c[0] = points[start].y2;
  min_c[1] = max_c[1] = points[start].x2;
  for (i = start + 1; i < end; i++)
  {
    if (points[i].y2 < min_c[0]) min_c[0] = points[i].y2;
    if (points[i].y2 > max_c[0]) max_c[0] = points[i].y2;
    if (points[i].x2 < min_c[1]) min_c[1] = points[i].x2;
    if (points[i].x2 > max_c[1]) max_c[1] = points[i].x2;
  }
}

static int kd3_build(Vertex3I *points, size_t start, size_t end, KDNode3 *nodes, int *node_count)
{
  size_t count;
  int node_idx, spread0, spread1, spread2;
  KDNode3 *node;
  size_t mid;

  node_idx = (*node_count)++;
  node = &nodes[node_idx];
  node->start = start;
  node->end = end;
  node->left = -1;
  node->right = -1;
  node->axis = -1;

  kd3_compute_bbox(points, start, end, node->min_c, node->max_c);
  count = end - start;
  if (count <= KD3_LEAF_SIZE)
    return node_idx;

  spread0 = node->max_c[0] - node->min_c[0];
  spread1 = node->max_c[1] - node->min_c[1];
  spread2 = node->max_c[2] - node->min_c[2];

  node->axis = 0;
  if (spread1 > spread0 && spread1 >= spread2)
    node->axis = 1;
  else if (spread2 > spread0 && spread2 > spread1)
    node->axis = 2;

  kd3_sort_axis = node->axis;
  qsort(points + start, count, sizeof(Vertex3I), compare_vertex3_axis);

  mid = start + count / 2;
  if (mid == start || mid == end)
    return node_idx;

  node->left = kd3_build(points, start, mid, nodes, node_count);
  node->right = kd3_build(points, mid, end, nodes, node_count);
  return node_idx;
}

static int kd2_build(Vertex2I *points, size_t start, size_t end, KDNode2 *nodes, int *node_count)
{
  size_t count;
  int node_idx, spread0, spread1;
  KDNode2 *node;
  size_t mid;

  node_idx = (*node_count)++;
  node = &nodes[node_idx];
  node->start = start;
  node->end = end;
  node->left = -1;
  node->right = -1;
  node->axis = -1;

  kd2_compute_bbox(points, start, end, node->min_c, node->max_c);
  count = end - start;
  if (count <= KD2_LEAF_SIZE)
    return node_idx;

  spread0 = node->max_c[0] - node->min_c[0];
  spread1 = node->max_c[1] - node->min_c[1];
  node->axis = (spread1 > spread0) ? 1 : 0;

  kd2_sort_axis = node->axis;
  qsort(points + start, count, sizeof(Vertex2I), compare_vertex2_axis);

  mid = start + count / 2;
  if (mid == start || mid == end)
    return node_idx;

  node->left = kd2_build(points, start, mid, nodes, node_count);
  node->right = kd2_build(points, mid, end, nodes, node_count);
  return node_idx;
}

static double kd3_upper_bound_sq(const KDNode3 *node, const Vertex3I *query, double sz_sq, double sy_sq, double sx_sq)
{
  int d0a = query->z2 - node->min_c[0];
  int d0b = query->z2 - node->max_c[0];
  int d1a = query->y2 - node->min_c[1];
  int d1b = query->y2 - node->max_c[1];
  int d2a = query->x2 - node->min_c[2];
  int d2b = query->x2 - node->max_c[2];
  int d0, d1, d2;

  if (d0a < 0) d0a = -d0a;
  if (d0b < 0) d0b = -d0b;
  if (d1a < 0) d1a = -d1a;
  if (d1b < 0) d1b = -d1b;
  if (d2a < 0) d2a = -d2a;
  if (d2b < 0) d2b = -d2b;
  d0 = (d0a > d0b) ? d0a : d0b;
  d1 = (d1a > d1b) ? d1a : d1b;
  d2 = (d2a > d2b) ? d2a : d2b;

  return (double)d0 * (double)d0 * sz_sq +
         (double)d1 * (double)d1 * sy_sq +
         (double)d2 * (double)d2 * sx_sq;
}

static double kd2_upper_bound_sq(const KDNode2 *node, const Vertex2I *query, double sy_sq, double sx_sq)
{
  int d0a = query->y2 - node->min_c[0];
  int d0b = query->y2 - node->max_c[0];
  int d1a = query->x2 - node->min_c[1];
  int d1b = query->x2 - node->max_c[1];
  int d0, d1;

  if (d0a < 0) d0a = -d0a;
  if (d0b < 0) d0b = -d0b;
  if (d1a < 0) d1a = -d1a;
  if (d1b < 0) d1b = -d1b;
  d0 = (d0a > d0b) ? d0a : d0b;
  d1 = (d1a > d1b) ? d1a : d1b;

  return (double)d0 * (double)d0 * sy_sq +
         (double)d1 * (double)d1 * sx_sq;
}

static double kd3_query_farthest_sq(
  const Vertex3I *points,
  const KDNode3 *nodes,
  int node_idx,
  const Vertex3I *query,
  double sz_sq,
  double sy_sq,
  double sx_sq,
  double best_sq
)
{
  const KDNode3 *node = &nodes[node_idx];
  size_t i;
  double ub = kd3_upper_bound_sq(node, query, sz_sq, sy_sq, sx_sq);
  if (ub <= best_sq)
    return best_sq;

  if (node->left < 0 || node->right < 0 || node->axis < 0)
  {
    for (i = node->start; i < node->end; i++)
    {
      int dz = query->z2 - points[i].z2;
      int dy = query->y2 - points[i].y2;
      int dx = query->x2 - points[i].x2;
      double d2 = (double)dz * (double)dz * sz_sq +
                  (double)dy * (double)dy * sy_sq +
                  (double)dx * (double)dx * sx_sq;
      if (d2 > best_sq)
        best_sq = d2;
    }
    return best_sq;
  }

  {
    int first = node->left;
    int second = node->right;
    double ub_first = kd3_upper_bound_sq(&nodes[first], query, sz_sq, sy_sq, sx_sq);
    double ub_second = kd3_upper_bound_sq(&nodes[second], query, sz_sq, sy_sq, sx_sq);
    if (ub_second > ub_first)
    {
      int tmp = first;
      first = second;
      second = tmp;
      {
        double td = ub_first;
        ub_first = ub_second;
        ub_second = td;
      }
    }
    if (ub_first > best_sq)
      best_sq = kd3_query_farthest_sq(points, nodes, first, query, sz_sq, sy_sq, sx_sq, best_sq);
    if (ub_second > best_sq)
      best_sq = kd3_query_farthest_sq(points, nodes, second, query, sz_sq, sy_sq, sx_sq, best_sq);
  }

  return best_sq;
}

static double kd2_query_farthest_sq(
  const Vertex2I *points,
  const KDNode2 *nodes,
  int node_idx,
  const Vertex2I *query,
  double sy_sq,
  double sx_sq,
  double best_sq
)
{
  const KDNode2 *node = &nodes[node_idx];
  size_t i;
  double ub = kd2_upper_bound_sq(node, query, sy_sq, sx_sq);
  if (ub <= best_sq)
    return best_sq;

  if (node->left < 0 || node->right < 0 || node->axis < 0)
  {
    for (i = node->start; i < node->end; i++)
    {
      int dy = query->y2 - points[i].y2;
      int dx = query->x2 - points[i].x2;
      double d2 = (double)dy * (double)dy * sy_sq +
                  (double)dx * (double)dx * sx_sq;
      if (d2 > best_sq)
        best_sq = d2;
    }
    return best_sq;
  }

  {
    int first = node->left;
    int second = node->right;
    double ub_first = kd2_upper_bound_sq(&nodes[first], query, sy_sq, sx_sq);
    double ub_second = kd2_upper_bound_sq(&nodes[second], query, sy_sq, sx_sq);
    if (ub_second > ub_first)
    {
      int tmp = first;
      first = second;
      second = tmp;
      {
        double td = ub_first;
        ub_first = ub_second;
        ub_second = td;
      }
    }
    if (ub_first > best_sq)
      best_sq = kd2_query_farthest_sq(points, nodes, first, query, sy_sq, sx_sq, best_sq);
    if (ub_second > best_sq)
      best_sq = kd2_query_farthest_sq(points, nodes, second, query, sy_sq, sx_sq, best_sq);
  }

  return best_sq;
}

static void brute_force_diameters3(const Vertex3I *points, size_t count, double sz_sq, double sy_sq, double sx_sq, double *diameters_sq)
{
  size_t i, j;
  diameters_sq[0] = 0.0;
  diameters_sq[1] = 0.0;
  diameters_sq[2] = 0.0;
  diameters_sq[3] = 0.0;
  for (i = 0; i < count; i++)
  {
    for (j = i + 1; j < count; j++)
    {
      int dz = points[i].z2 - points[j].z2;
      int dy = points[i].y2 - points[j].y2;
      int dx = points[i].x2 - points[j].x2;
      double d2 = (double)dz * (double)dz * sz_sq +
                  (double)dy * (double)dy * sy_sq +
                  (double)dx * (double)dx * sx_sq;
      if (dz == 0 && d2 > diameters_sq[0]) diameters_sq[0] = d2;
      if (dy == 0 && d2 > diameters_sq[1]) diameters_sq[1] = d2;
      if (dx == 0 && d2 > diameters_sq[2]) diameters_sq[2] = d2;
      if (d2 > diameters_sq[3]) diameters_sq[3] = d2;
    }
  }
}

static double max_sq_group_same_z(Vertex3I *points, size_t count, double sy_sq, double sx_sq)
{
  size_t i, j, start, end;
  double best = 0.0;
  qsort(points, count, sizeof(Vertex3I), compare_vertex3_z_yx);
  start = 0;
  while (start < count)
  {
    end = start + 1;
    while (end < count && points[end].z2 == points[start].z2) end++;
    for (i = start; i < end; i++)
    {
      for (j = i + 1; j < end; j++)
      {
        int dy = points[i].y2 - points[j].y2;
        int dx = points[i].x2 - points[j].x2;
        double d2 = (double)dy * (double)dy * sy_sq +
                    (double)dx * (double)dx * sx_sq;
        if (d2 > best) best = d2;
      }
    }
    start = end;
  }
  return best;
}

static double max_sq_group_same_y(Vertex3I *points, size_t count, double sz_sq, double sx_sq)
{
  size_t i, j, start, end;
  double best = 0.0;
  qsort(points, count, sizeof(Vertex3I), compare_vertex3_y_zx);
  start = 0;
  while (start < count)
  {
    end = start + 1;
    while (end < count && points[end].y2 == points[start].y2) end++;
    for (i = start; i < end; i++)
    {
      for (j = i + 1; j < end; j++)
      {
        int dz = points[i].z2 - points[j].z2;
        int dx = points[i].x2 - points[j].x2;
        double d2 = (double)dz * (double)dz * sz_sq +
                    (double)dx * (double)dx * sx_sq;
        if (d2 > best) best = d2;
      }
    }
    start = end;
  }
  return best;
}

static double max_sq_group_same_x(Vertex3I *points, size_t count, double sz_sq, double sy_sq)
{
  size_t i, j, start, end;
  double best = 0.0;
  qsort(points, count, sizeof(Vertex3I), compare_vertex3_x_zy);
  start = 0;
  while (start < count)
  {
    end = start + 1;
    while (end < count && points[end].x2 == points[start].x2) end++;
    for (i = start; i < end; i++)
    {
      for (j = i + 1; j < end; j++)
      {
        int dz = points[i].z2 - points[j].z2;
        int dy = points[i].y2 - points[j].y2;
        double d2 = (double)dz * (double)dz * sz_sq +
                    (double)dy * (double)dy * sy_sq;
        if (d2 > best) best = d2;
      }
    }
    start = end;
  }
  return best;
}

// Declare the look-up tables, these are filled at the bottom of this code file.
static const int gridAngles[8][3];
//static const int edgeTable[128];  // Not needed in this implementation
static const int triTable[128][16];
static const double vertList[12][3];

int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters)
{
  int iz, iy, ix, i, t, d;  // iterator indices
  unsigned char cube_idx;  // cube identifier, 8 bits signifying which corners of the cube belong to the segmentation
  int a_idx;  // Angle index (8 'angles', one pointing to each corner of the marching cube

  static const int points_edges[2][3] = {{6, 4, 3}, {6, 7, 11}};
  size_t v_idx = 0;
  size_t v_max = 0;
  Vertex3I *vertices;

  double sum;
  double a[3], b[3], c[3], ab[3];  // 3 points of the triangle, and the cross product vector
  int sign_correction;

  *surfaceArea = 0;  // Total surface area
  *volume = 0;  // Total volume

  // create a stack to hold the found vertices. For each cube, a maximum of 3 vertices are stored (with x, y and z
  // coordinates). This prevents double storing of the vertices.
  v_max = (size[0] - 1) * (size[1] - 1) * (size[2] - 1) * 3;
  vertices = (Vertex3I *)calloc(v_max, sizeof(Vertex3I));
  if (!vertices)
    return 1;

  // Iterate over all voxels, do not include last voxels in the three dimensions, as the cube includes voxels at pos +1
  for (iz = 0; iz < (size[0] - 1); iz++)
  {
    for (iy = 0; iy < (size[1] - 1); iy++)
    {
      for (ix = 0; ix < (size[2] - 1); ix++)
      {
        /* Get current cube_idx by analyzing each point of the current cube
        * O - X
        * |\
        * Y Z
        *           v0
        *  p0 ------------ p1
        *   |\             |\
        *   | \ v3         | \ v1
        * v8|  \      v2   |v9\
        *   |  p3 ------------ p2
        *   |   |  v4      |   |
        *  p4 --|--------- p5  |
        *    \  |v11        \  |v10
        *  v7 \ |          v5\ |
        *      \|             \|
        *      p7 ------------ p6
        *             v6
        */
        cube_idx = 0;
        for (a_idx = 0; a_idx < 8; a_idx++)
        {
          i = (iz + gridAngles[a_idx][0]) * strides[0] +
              (iy + gridAngles[a_idx][1]) * strides[1] +
              (ix + gridAngles[a_idx][2]) * strides[2];

          if (mask[i])
            cube_idx |= (1 << a_idx);
        }

        // Isosurface is symmetrical around the midpoint, flip the number if > 128
        // This enables look-up tables to be 1/2 the size.
        // However, the sign for the volume then needs to be flipped too.
        if (cube_idx & 0x80)
        {
          cube_idx ^= 0xff;
          sign_correction = -1;
        }
        else
          sign_correction = 1;

        // ************************
        // Store vertices for diameter calculation
        // ************************

        // check if there are vertices on edges 6, 7 and 11
        // Because of the symmetry around the midpoint and the flip if cube_idx > 128, the 8th point will never appear
        // as segmented at this point. Therefore, to check if there are vertices on the adjacent edges (6, 7 and 11),
        // one only needs to check if the corresponding points (7th, 5th and 4th, respectively) are segmented.
        if (v_idx + 3 > v_max) // Overflow!
        {
          free(vertices);
          return 1;
        }

        for (t = 0; t < 3; t++)
        {
          if (cube_idx & (1 << points_edges[0][t]))
          {
            int edge = points_edges[1][t];
            vertices[v_idx].z2 = 2 * iz + (int)lrint(2.0 * vertList[edge][0]);
            vertices[v_idx].y2 = 2 * iy + (int)lrint(2.0 * vertList[edge][1]);
            vertices[v_idx].x2 = 2 * ix + (int)lrint(2.0 * vertList[edge][2]);
            v_idx++;
          }
        }

        // Exclude cubes entirely outside or inside the segmentation (cube_idx = 0).
        if (cube_idx == 0)
          continue;

        // Process all triangles for this cube
        t = 0;
        while (triTable[cube_idx][t*3] >= 0) // Exit loop when no more triangles are present (element at index = -1)
        {
          a[0] = b[0] = c[0] = iz;
          a[1] = b[1] = c[1] = iy;
          a[2] = b[2] = c[2] = ix;
          for (d = 0; d < 3; d++)
          {
              a[d] += vertList[triTable[cube_idx][t*3]][d];
              b[d] += vertList[triTable[cube_idx][t*3 + 1]][d];
              c[d] += vertList[triTable[cube_idx][t*3 + 2]][d];
              // Factor in the spacing
              a[d] *= spacing[d];
              b[d] *= spacing[d];
              c[d] *= spacing[d];
          }

          // ************************
          // Calculate volume
          // ************************

          // Calculate the cross product
          ab[0] = (a[1] * b[2]) - (b[1] * a[2]);
          ab[1] = (a[2] * b[0]) - (b[2] * a[0]);
          ab[2] = (a[0] * b[1]) - (b[0] * a[1]);

          // Calculate the dot-product and add it to the volume total. The division by 6 is performed at the end.
          *volume += sign_correction * (ab[0] * c[0] + ab[1] * c[1] + ab[2] * c[2]);

          // ************************
          // Calculate surface area
          // ************************

          // Compute the surface, which is equal to 1/2 magnitude of the cross product, where
          // The magnitude is obtained by calculating the euclidean distance between (0, 0, 0)
          // and the location of c
          for (d = 0; d < 3; d++)
          {
            a[d] -= c[d];
            b[d] -= c[d];
          }

          // Compute the cross-product
          ab[0] = (a[1] * b[2]) - (b[1] * a[2]);
          ab[1] = (a[2] * b[0]) - (b[2] * a[0]);
          ab[2] = (a[0] * b[1]) - (b[0] * a[1]);

          // Get the euclidean distance by computing the square and then the square root of the sum.
          ab[0] = ab[0] * ab[0];
          ab[1] = ab[1] * ab[1];
          ab[2] = ab[2] * ab[2];

          sum = ab[0] + ab[1] + ab[2];
          sum = sqrt(sum);

          // multiply by 0.5 (1/2 the magnitude of the cross product)
          sum = 0.5 * sum;

          // Add the surface area of the face to the grand total.
          *surfaceArea += sum;
          t++;
        }
      }
    }
  }
  *volume = *volume / 6;

  // ************************
  // Calculate Diameters using found vertices
  // ************************
  calculate_meshDiameter(vertices, v_idx, spacing, diameters);
  free(vertices);
  return 0;
}

void calculate_meshDiameter(const Vertex3I *points, size_t count, const double *spacing, double *diameters)
{
  Vertex3I *unique_points = NULL;
  Vertex3I *tmp_points = NULL;
  KDNode3 *nodes = NULL;
  double sz = 0.5 * spacing[0];
  double sy = 0.5 * spacing[1];
  double sx = 0.5 * spacing[2];
  double sz_sq = sz * sz;
  double sy_sq = sy * sy;
  double sx_sq = sx * sx;
  double diam_sq[4] = {0.0, 0.0, 0.0, 0.0};
  size_t unique_count;
  size_t i;

  diameters[0] = 0.0;
  diameters[1] = 0.0;
  diameters[2] = 0.0;
  diameters[3] = 0.0;
  if (count < 2)
    return;

  unique_points = (Vertex3I *)malloc(count * sizeof(Vertex3I));
  if (!unique_points)
  {
    brute_force_diameters3(points, count, sz_sq, sy_sq, sx_sq, diam_sq);
    diameters[0] = sqrt(diam_sq[0]);
    diameters[1] = sqrt(diam_sq[1]);
    diameters[2] = sqrt(diam_sq[2]);
    diameters[3] = sqrt(diam_sq[3]);
    return;
  }
  memcpy(unique_points, points, count * sizeof(Vertex3I));
  unique_count = dedup_vertex3(unique_points, count);
  if (unique_count < 2)
  {
    free(unique_points);
    return;
  }

  nodes = (KDNode3 *)malloc((2 * unique_count + 1) * sizeof(KDNode3));
  if (!nodes)
  {
    brute_force_diameters3(unique_points, unique_count, sz_sq, sy_sq, sx_sq, diam_sq);
    diameters[0] = sqrt(diam_sq[0]);
    diameters[1] = sqrt(diam_sq[1]);
    diameters[2] = sqrt(diam_sq[2]);
    diameters[3] = sqrt(diam_sq[3]);
    free(unique_points);
    return;
  }

  {
    int node_count = 0;
    kd3_build(unique_points, 0, unique_count, nodes, &node_count);
    diam_sq[3] = 0.0;
    for (i = 0; i < unique_count; i++)
    {
      diam_sq[3] = kd3_query_farthest_sq(
        unique_points, nodes, 0, &unique_points[i], sz_sq, sy_sq, sx_sq, diam_sq[3]
      );
    }
  }
  free(nodes);

  tmp_points = (Vertex3I *)malloc(unique_count * sizeof(Vertex3I));
  if (!tmp_points)
  {
    brute_force_diameters3(unique_points, unique_count, sz_sq, sy_sq, sx_sq, diam_sq);
    diameters[0] = sqrt(diam_sq[0]);
    diameters[1] = sqrt(diam_sq[1]);
    diameters[2] = sqrt(diam_sq[2]);
    diameters[3] = sqrt(diam_sq[3]);
    free(unique_points);
    return;
  }

  memcpy(tmp_points, unique_points, unique_count * sizeof(Vertex3I));
  diam_sq[0] = max_sq_group_same_z(tmp_points, unique_count, sy_sq, sx_sq);
  memcpy(tmp_points, unique_points, unique_count * sizeof(Vertex3I));
  diam_sq[1] = max_sq_group_same_y(tmp_points, unique_count, sz_sq, sx_sq);
  memcpy(tmp_points, unique_points, unique_count * sizeof(Vertex3I));
  diam_sq[2] = max_sq_group_same_x(tmp_points, unique_count, sz_sq, sy_sq);

  free(tmp_points);
  free(unique_points);

  diameters[0] = sqrt(diam_sq[0]);
  diameters[1] = sqrt(diam_sq[1]);
  diameters[2] = sqrt(diam_sq[2]);
  diameters[3] = sqrt(diam_sq[3]);
}

// gridAngles define the 8 corners of the marching cube, relative to the origin of the cube
static const int gridAngles[8][3] = { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 1 }, {0, 1, 0}, { 1, 0, 0 }, {1, 0, 1 }, { 1, 1, 1 }, { 1, 1, 0 } };

// edgeTable defines which edges contain intersection points, for which the exact intersection point has to be
// interpolated. However, as the intersection point is always 0.5, this can be defined beforehand, and this table is not
// needed
/*static const int edgeTable[128] = {
  0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0
};*/

// triTable defines which triangles (defined by their points as defined in vertList) are present in the cube.
// The first dimension indicates the specific cube to look up, the second dimension contains sets of 3 points (1 for
// each triangle), with the elements set to -1 after all triangles have been defined (max. no of triangles: 5)
static const int triTable[128][16] = {
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 8, 3, 1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 0, 0, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 1, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
  { 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 10, 1, 0, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 0, 3, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 8, 10, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 4, 3, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 4, 1, 4, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 2, 10, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 2, 10, 9, 0, 2, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 10, 9, 2, 9, 7, 2, 7, 3, 4, 7, 9, -1, -1, -1, -1 },
  { 4, 7, 8, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 4, 7, 8, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 4, 7, 11, 9, 4, 11, 11, 2, 9, 1, 9, 2, -1, -1, -1, -1 },
  { 3, 10, 1, 11, 10, 3, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 10, 1, 1, 4, 11, 1, 0, 4, 4, 7, 11, -1, -1, -1, -1 },
  { 4, 7, 8, 9, 0, 3, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1 },
  { 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 5, 4, 0, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 8, 3, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 10, 5, 5, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 10, 5, 3, 2, 5, 3, 5, 4, 8, 3, 4, -1, -1, -1, -1 },
  { 9, 5, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 0, 0, 8, 11, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 5, 4, 0, 1, 5, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 1, 5, 2, 5, 8, 11, 2, 8, 5, 4, 8, -1, -1, -1, -1 },
  { 3, 10, 1, 11, 10, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
  { 5, 4, 9, 10, 1, 0, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1 },
  { 5, 4, 0, 5, 0, 11, 10, 5, 11, 11, 0, 3, -1, -1, -1, -1 },
  { 5, 4, 8, 10, 5, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1 },
  { 7, 8, 9, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 3, 0, 9, 5, 3, 9, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
  { 7, 8, 0, 0, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 5, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 7, 8, 9, 5, 7, 9, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 2, 10, 3, 0, 9, 3, 9, 5, 3, 5, 7, -1, -1, -1, -1 },
  { 0, 2, 8, 8, 2, 5, 8, 5, 7, 2, 10, 5, -1, -1, -1, -1 },
  { 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
  { 5, 7, 9, 7, 8, 9, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 5, 7, 9, 9, 7, 2, 2, 0, 9, 11, 2, 7, -1, -1, -1, -1 },
  { 11, 2, 3, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
  { 2, 1, 11, 1, 7, 11, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
  { 7, 8, 9, 5, 7, 9, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1 },
  { 5, 7, 0, 5, 0, 9, 7, 11, 0, 10, 1, 0, 11, 10, 0, -1 },
  { 11, 10, 0, 0, 3, 11, 10, 5, 0, 0, 7, 8, 5, 7, 0, -1 },
  { 11, 10, 5, 5, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 8, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 8, 3, 1, 9, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 6, 5, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 6, 5, 1, 2, 6, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
  { 5, 9, 8, 5, 8, 2, 5, 2, 6, 8, 3, 2, -1, -1, -1, -1 },
  { 11, 2, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 0, 0, 8, 11, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 11, 2, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 1, 1, 9, 11, 9, 8, 11, 6, 5, 10, -1, -1, -1, -1 },
  { 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
  { 6, 3, 11, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
  { 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
  { 4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 4, 3, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 0, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 9, 4, 1, 4, 7, 1, 7, 3, 6, 5, 10, -1, -1, -1, -1 },
  { 4, 7, 8, 1, 6, 5, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 4, 3, 4, 7, 3, 1, 6, 5, 1, 2, 6, -1, -1, -1, -1 },
  { 4, 7, 8, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1 },
  { 7, 3, 9, 4, 7, 9, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
  { 11, 2, 3, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 4, 7, 11, 2, 4, 2, 0, 4, 6, 5, 10, -1, -1, -1, -1 },
  { 1, 9, 0, 11, 2, 3, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1 },
  { 4, 7, 11, 9, 4, 11, 11, 2, 9, 1, 9, 2, 6, 5, 10, -1 },
  { 4, 7, 8, 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1 },
  { 5, 1, 11, 5, 11, 6, 1, 0, 11, 4, 7, 11, 0, 4, 11, -1 },
  { 4, 7, 8, 6, 3, 11, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1 },
  { 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
  { 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 10, 4, 9, 6, 4, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 1, 10, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 8, 3, 1, 6, 8, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
  { 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 4, 9, 1, 2, 4, 2, 6, 4, 0, 8, 3, -1, -1, -1, -1 },
  { 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 3, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 0, 0, 8, 11, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1 },
  { 11, 2, 3, 0, 1, 10, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1 },
  { 6, 4, 1, 6, 1, 10, 1, 4, 8, 11, 2, 1, 1, 8, 11, -1 },
  { 9, 6, 4, 3, 6, 9, 1, 3, 9, 11, 6, 3, -1, -1, -1, -1 },
  { 1, 8, 11, 0, 8, 1, 11, 6, 1, 1, 4, 9, 6, 4, 1, -1 },
  { 6, 3, 11, 0, 3, 6, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
  { 8, 6, 4, 6, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 6, 7, 10, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
  { 6, 7, 10, 0, 10, 7, 0, 9, 10, 0, 7, 3, -1, -1, -1, -1 },
  { 6, 7, 10, 1, 10, 7, 1, 7, 8, 0, 1, 8, -1, -1, -1, -1 },
  { 6, 7, 10, 1, 10, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
  { 1, 2, 6, 1, 6, 8, 1, 8, 9, 6, 7, 8, -1, -1, -1, -1 },
  { 2, 6, 9, 1, 2, 9, 6, 7, 9, 3, 0, 9, 7, 3, 9, -1 },
  { 0, 7, 8, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
  { 2, 7, 3, 2, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 11, 2, 3, 6, 7, 10, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1 },
  { 2, 0, 7, 11, 2, 7, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
  { 6, 7, 10, 1, 10, 7, 1, 7, 8, 0, 1, 8, 11, 2, 3, -1 },
  { 11, 2, 1, 1, 7, 11, 10, 6, 1, 1, 6, 7, -1, -1, -1, -1 },
  { 8, 9, 6, 6, 7, 8, 1, 6, 9, 11, 6, 3, 1, 3, 6, -1 },
  { 0, 9, 1, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
  { 0, 7, 8, 7, 0, 6, 0, 3, 11, 11, 6, 0, -1, -1, -1, -1 },
  { 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

// Vertlist represents the location of some point somewhere on an edge of the cube, relative to the origin (0, 0, 0).
// As the points on the cube are always either 0 or 1 (masked/not-masked) that other point is always halfway.
// Therefore, vertlist is constant and can be defined static (only works when the intersection point is constant
// (in this case the intersection point is always 0.5). The edge represented is defined by the gridAngle points as follows:
// { { 1, 0 }, { 2, 1 }, { 3, 2 }, { 3, 0 },
//   { 5, 4 }, { 6, 5 }, { 7, 6 }, { 7, 4 },
//   { 4, 0 }, { 5, 1 }, { 6, 2 }, { 7, 0 } }
static const double vertList[12][3] = { { 0, 0, 0.5 }, { 0, 0.5, 1 }, { 0, 1, 0.5 }, { 0, 0.5, 0 },
                                        { 1, 0, 0.5 }, { 1, 0.5, 1 }, { 1, 1, 0.5 }, { 1, 0.5, 0 },
                                        { 0.5, 0, 0 }, { 0.5, 0, 1 }, { 0.5, 1, 1 }, { 0.5, 1, 0 } };


// **************************************************

// 2D Shape calculation

// **************************************************

// Declare the look-up tables, these are filled at the bottom of this code file.
static const int gridAngles2D[4][2];
static const int lineTable2D[16][5];
static const double vertList2D[4][2];

int calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter)
{
  int iy, ix, i, t, d;  // iterator indices
  unsigned char square_idx;  // cube identifier, 4 bits signifying which corners of the cube belong to the segmentation
  int a_idx;  // Angle index (4 'angles', one pointing to each corner of the marching cube

  static const int points_edges[2][2] = {{0, 2}, {3, 2}};
  size_t v_idx = 0;
  size_t v_max = 0;
  Vertex2I *vertices;

  double sum;
  double a[2], b[2];  // 2 points of the line

  *perimeter = 0;  // Total perimeter
  *surface = 0;  // Total surface

  // create a stack to hold the found vertices. For each square, a maximum of 2 vertices are stored (with x and y)
  // coordinates). This prevents double storing of the vertices.
  v_max = (size[0] - 1) * (size[1] - 1) * 2;
  vertices = (Vertex2I *)calloc(v_max, sizeof(Vertex2I));
  if (!vertices)
    return 1;

  // Iterate over all pixels, do not include last voxels in the three dimensions, as the cube includes voxels at pos +1
  for (iy = 0; iy < (size[0] - 1); iy++)
  {
    for (ix = 0; ix < (size[1] - 1); ix++)
    {

      /* Get current square_idx by analyzing each point of the current square (origin is in left-upper corner)
      *  O - X
      *  |
      *  Y
      *         v0
      *   p0 ------- p1
      *    |         |
      * v3 |         | v1
      *    |         |
      *   p3 ------- p2
      *         v2
      */

      square_idx = 0;
      for (a_idx = 0; a_idx < 4; a_idx++)
      {
        i = (iy + gridAngles2D[a_idx][0]) * strides[0] +
            (ix + gridAngles2D[a_idx][1]) * strides[1];

        if (mask[i])
          square_idx |= (1 << a_idx);
      }

      // Exclude squares entirely outside or inside the segmentation (square_idx = 0 or 0xF = B1111).
      if (square_idx == 0 || square_idx == 0xF)
        continue;

      // Process all lines for this square
      t = 0;
      while (lineTable2D[square_idx][t*2] >= 0) // Exit loop when no more lines are present (element at index = -1)
      {
        a[0] = b[0] = iy;
        a[1] = b[1] = ix;
        for (d = 0; d < 2; d++)
        {
            a[d] += vertList2D[lineTable2D[square_idx][t*2]][d];
            b[d] += vertList2D[lineTable2D[square_idx][t*2 + 1]][d];
            // Factor in the spacing
            a[d] *= spacing[d];
            b[d] *= spacing[d];
        }

        // ************************
        // Calculate Surface
        // ************************

        // Calculate the cross product. Because for both vectors, z = 0, only the last term need be calculated
        // The surface of the triangle is only 1/2 the magnitude of this result, but the division by 2 is done on the
        // final sum.
        *surface += (a[0] * b[1]) - (b[0] * a[1]);

        // ************************
        // Calculate perimeter
        // ************************

        // Compute the euclidean distance between points a and b.
        // Add the result to the grand total, as the perimeter is the sum of
        // all line lengths.
        for (d = 0; d < 2; d++)
        {
          a[d] -= b[d];

          // Get the euclidean distance by computing the square...
          a[d] = a[d] * a[d];
        }

        // ... and then the square root of the sum.
        sum = a[0] + a[1];
        sum = sqrt(sum);

        // Add the length of the line to the grand total.
        *perimeter += sum;
        t++;
      }

      // ************************
      // Store vertices for diameter calculation
      // ************************

      // check if there are vertices on edges 3 and 2
      // Because of the symmetry around the midpoint and the flip if cube_idx > 0xF, the 4th point will never appear
      // as segmented at this point. Therefore, to check if there are vertices on the adjacent edges (3 and 2),
      // one only needs to check if the corresponding points (0 and 2, respectively) are segmented.
      if (v_idx + 2 > v_max) // Overflow!
      {
        free(vertices);
        return 1;
      }

      if (square_idx > 7)
        square_idx = square_idx ^ 0xF;  // Flip the square index

      for (t = 0; t < 2; t++)
      {
        if (square_idx & (1 << points_edges[0][t]))
        {
          int edge = points_edges[1][t];
          vertices[v_idx].y2 = 2 * iy + (int)lrint(2.0 * vertList2D[edge][0]);
          vertices[v_idx].x2 = 2 * ix + (int)lrint(2.0 * vertList2D[edge][1]);
          v_idx++;
        }
      }
    }
  }
  // The surface area of the triangle is 1/2 the magnitude of the cross product.
  *surface = *surface / 2;

  // ************************
  // Calculate Diameters using found vertices
  // ************************
  *diameter = calculate_meshDiameter2D(vertices, v_idx, spacing);
  free(vertices);
  return 0;
}


double calculate_meshDiameter2D(const Vertex2I *points, size_t count, const double *spacing)
{
  Vertex2I *unique_points;
  KDNode2 *nodes;
  double sy = 0.5 * spacing[0];
  double sx = 0.5 * spacing[1];
  double sy_sq = sy * sy;
  double sx_sq = sx * sx;
  double best_sq = 0.0;
  size_t unique_count;
  size_t i, j;

  if (count < 2)
    return 0.0;

  unique_points = (Vertex2I *)malloc(count * sizeof(Vertex2I));
  if (!unique_points)
  {
    for (i = 0; i < count; i++)
    {
      for (j = i + 1; j < count; j++)
      {
        int dy = points[i].y2 - points[j].y2;
        int dx = points[i].x2 - points[j].x2;
        double d2 = (double)dy * (double)dy * sy_sq +
                    (double)dx * (double)dx * sx_sq;
        if (d2 > best_sq) best_sq = d2;
      }
    }
    return sqrt(best_sq);
  }

  memcpy(unique_points, points, count * sizeof(Vertex2I));
  unique_count = dedup_vertex2(unique_points, count);
  if (unique_count < 2)
  {
    free(unique_points);
    return 0.0;
  }

  nodes = (KDNode2 *)malloc((2 * unique_count + 1) * sizeof(KDNode2));
  if (!nodes)
  {
    for (i = 0; i < unique_count; i++)
    {
      for (j = i + 1; j < unique_count; j++)
      {
        int dy = unique_points[i].y2 - unique_points[j].y2;
        int dx = unique_points[i].x2 - unique_points[j].x2;
        double d2 = (double)dy * (double)dy * sy_sq +
                    (double)dx * (double)dx * sx_sq;
        if (d2 > best_sq) best_sq = d2;
      }
    }
    free(unique_points);
    return sqrt(best_sq);
  }

  {
    int node_count = 0;
    kd2_build(unique_points, 0, unique_count, nodes, &node_count);
    for (i = 0; i < unique_count; i++)
    {
      best_sq = kd2_query_farthest_sq(unique_points, nodes, 0, &unique_points[i], sy_sq, sx_sq, best_sq);
    }
  }
  free(nodes);
  free(unique_points);
  return sqrt(best_sq);
}


static const int gridAngles2D[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 } };


static const int lineTable2D[16][5] = {
  { -1, -1, -1, -1, -1},
  {  3,  0, -1, -1, -1},
  {  0,  1, -1, -1, -1},
  {  3,  1, -1, -1, -1},
  {  1,  2, -1, -1, -1},
  {  1,  2,  3,  0, -1},
  {  0,  2, -1, -1, -1},
  {  3,  2, -1, -1, -1},
  {  2,  3, -1, -1, -1},
  {  2,  0, -1, -1, -1},
  {  0,  1,  2,  3, -1},
  {  2,  1, -1, -1, -1},
  {  1,  3, -1, -1, -1},
  {  1,  0, -1, -1, -1},
  {  0,  3, -1, -1, -1},
  { -1, -1, -1, -1, -1},
};

static const double vertList2D[4][2] = { { 0, 0.5 }, { 0.5, 1 }, { 1, 0.5 }, { 0.5, 0 }};
