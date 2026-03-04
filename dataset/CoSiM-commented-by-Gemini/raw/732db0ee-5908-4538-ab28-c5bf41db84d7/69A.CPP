/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * This program determines if a body is in equilibrium under the influence of
 * multiple forces. It reads 'n' 3D force vectors, calculates their vector sum,
 * and checks if the resultant vector is the zero vector.
 *
 * @algorithm Vector component summation.
 * @time_complexity O(N), where N is the number of force vectors, as it iterates
 *                  through them once to sum the components.
 * @space_complexity O(1), as the space used is constant regardless of the
 *                   number of vectors.
 */
/*
 * 	User: Isanchez_Aguilar
 * 	Problem: CodeForces 69A - Young Physicist
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  // 'ans' vector will store the sum of x, y, and z components respectively.
  vector<int> ans(3);

  /**
   * @brief Reads each vector and accumulates the component sums.
   * This loop iterates 'n' times, reading a 3D vector (x, y, z) in each
   * iteration and adding its components to the corresponding element of
   * the 'ans' vector.
   */
  for (int i = 0; i < n; ++i) {
    int x, y, z;
    cin >> x >> y >> z;

    ans[0] += x;
    ans[1] += y;
    ans[2] += z;
  }

  /**
   * @brief Checks if the system is in equilibrium.
   * The system is in equilibrium if the net force is zero, which means the
   * sum of components along each axis (x, y, and z) must be zero.
   * `std::count` is used here to check if all three elements in the 'ans'
   * vector are zero.
   */
  if (count(begin(ans), end(ans), 0) == 3)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}
