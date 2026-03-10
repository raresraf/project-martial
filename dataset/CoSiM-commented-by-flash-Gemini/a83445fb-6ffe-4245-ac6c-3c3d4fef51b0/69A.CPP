/**
 * @file 69A.CPP
 * @brief This program solves the "Young Physicist" problem (Codeforces 69A).
 *   It determines if a body is in equilibrium given a set of `n` force vectors.
 *   A body is in equilibrium if the sum of all forces acting on it is zero,
 *   meaning the sum of their x-components, y-components, and z-components must all be zero.
 *   Problem description: https://codeforces.com/problemset/problem/69/A
 * * Algorithm: Vector sum.
 *   It reads `n` force vectors (each with x, y, z components) into a 2D vector structure.
 *   Then, it sums up all corresponding components separately. The body is in equilibrium
 *   if and only if all three component sums (sum_x, sum_y, sum_z) are zero.
 * Time Complexity: O(N), where N is the number of force vectors, as it iterates through them once for input and once for summation.
 * Space Complexity: O(N) to store the `n` force vectors in `std::vector<std::vector<int>> t`.
 */
#include <algorithm> // Not directly used but often included in competitive programming.
#include <iostream>  // Required for std::cin, std::cout.
#include <iterator>  // Not directly used but often included in competitive programming.
#include <vector>    // Required for std::vector.
using namespace std;

/**
 * @brief Main function to determine if a body is in equilibrium.
 * Functional Utility: Reads the number of force vectors and their components,
 *                     calculates the total force in each dimension, and prints
 *                     "YES" if the body is in equilibrium, "NO" otherwise.
 */
int main() {
  int n; // Stores the number of force vectors.
  // Block Logic: Reads the total number of force vectors, `n`.
  cin >> n;
  vector<int> v(3); // Temporary vector to read x, y, z components of a single force vector.
  vector<vector<int>> t; // Stores all `n` force vectors. `t` is a vector of vectors.

  /**
   * Block Logic: Reads `n` force vectors from input and stores them in `t`.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: After this loop, `t` contains `n` vectors, each with 3 components.
   */
  for (int i = 0; i < n; i++) {
    // v.clear(); // This line is commented out; `v` is reused without clearing.
    // Inner loop reads 3 components for the current force vector.
    for (int j = 0; j < 3; j++) {
      cin >> v[j];
    }
    t.push_back(v); // Adds the read force vector (stored in `v`) to the collection `t`.
  }

  // iterator<vector<int>>::it; // Commented out, likely debugging remnant or unused declaration.
  // `a`, `b`, `c`: Accumulators for the sum of x, y, and z components across all force vectors.
  int a = 0, b = 0, c = 0;
  /**
   * Block Logic: Iterates through the stored force vectors (`t`) to sum their components.
   * Pre-condition: `t` is populated with `n` force vectors.
   * Invariant: After this loop, `a`, `b`, `c` will contain the total sums of x, y, and z components respectively.
   */
  for (int i = 0; i < t.size(); i++) {
    for (int j = 0; j < t[i].size(); j++) {
      // cout<<t[i][j]<<" "; // Commented out, likely debugging print.
      // Conditional Logic: Sums components based on their index (0 for x, 1 for y, 2 for z).
      if (j == 0)
        a += t[i][j]; // Sums x-components.
      else if (j == 1)
        b += t[i][j]; // Sums y-components.
      else
        c += t[i][j]; // Sums z-components.
    }
    // cout<<"\n"; // Commented out, likely debugging print.
  }
  // cout<<a<<" "<<b<<" "<<c<<"\n"; // Commented out, likely debugging print.

  /**
   * Conditional Logic: Checks if the body is in equilibrium.
   * A body is in equilibrium if the sum of all x, y, and z components is zero.
   * If `a`, `b`, and `c` are all zero, it prints "YES". Otherwise, it prints "NO".
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}