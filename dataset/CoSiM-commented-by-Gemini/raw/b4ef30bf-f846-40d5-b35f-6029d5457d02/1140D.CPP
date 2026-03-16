/**
 * @file 1140D.CPP
 * @brief Solves the "Minimum Triangulation" problem.
 * @url https://codeforces.com/problemset/problem/1140/D
 *
 * @details
 * The problem asks for the minimum cost to triangulate a regular n-sided polygon.
 * The vertices of the polygon are labeled 1 to n in clockwise order. The cost of a
 * triangulation is the sum of the weights of all triangles in it, where the weight
 * of a triangle is the product of its vertices' labels.
 *
 * @algorithm
 * The optimal strategy to minimize the total weight is to form a "fan" triangulation,
 * where all triangles share a common vertex (e.g., vertex 1). This triangulation
 * consists of triangles (1, 2, 3), (1, 3, 4), ..., (1, n-1, n).
 * The total cost is the sum of the weights of these triangles:
 * 1*2*3 + 1*3*4 + ... + 1*(n-1)*n.
 * This simplifies to the sum of `i * (i+1)` for `i` from 2 to `n-1`.
 * The program calculates this sum iteratively.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vertices of the polygon,
 * due to the single loop from 2 to n-1.
 * Space Complexity: O(1).
 */
#include <iostream>

void answer(unsigned v) { std::cout << v << '
'; }

void solve(unsigned n) {
  unsigned s = 0;
  
  /**
   * @brief Calculate the sum of weights of the triangulation.
   * This loop implements the summation `s = sum_{x=2}^{n-1} x * (x + 1)`.
   */
  for (unsigned x = 2; x < n; ++x)
    s += x * (x + 1);

  answer(s);
}

int main() {
  unsigned n;
  std::cin >> n;

  solve(n);

  return 0;
}
