/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" programming problem.
 * @details This program determines if a body is in equilibrium under the influence of several forces.
 * According to Newton's first law, a body is in equilibrium if the vector sum of all forces
 * acting upon it is zero. For 3D vectors, this means the sum of all x-components, the sum of all
 * y-components, and the sum of all z-components must each be zero. This program reads a set of
 * force vectors, calculates the sum of their components along each axis, and checks if all
 * three sums are zero.
 *
 * @algorithm Vector Summation
 * @complexity
 * Time Complexity: O(N), where N is the number of force vectors. The program iterates through the
 * vectors twice: once to read them and once to sum their components.
 * Space Complexity: O(N), due to storing the components of N vectors in arrays. Note that this
 * could be optimized to O(1) by summing the components as they are read, avoiding storage.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  
  // Declaration of Variable-Length Arrays (VLA) to store the vector components.
  // Note: VLA is a non-standard C++ feature, but it is supported as an extension by some compilers like GCC.
  int x[n], y[n], z[n];
  
  // Block Logic: Read all N force vectors from the input.
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  
  int sx = 0, sy = 0, sz = 0;
  // Block Logic: Calculate the sum of components for each axis.
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sz += z[i];
  }
  
  // Block Invariant: Check for equilibrium. The condition `!sx` is true if and only if `sx` is 0.
  // The body is in equilibrium if and only if the sum of forces along all three axes is zero.
  if (!sx && !sy && !sz)
    cout << "YES"
         << "\n";
  else
    cout << "NO"
         << "\n";
  return 0;
}