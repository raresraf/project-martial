/**
 * @file 69A.CPP
 * @brief Solution for a competitive programming problem about vector equilibrium.
 *
 * This program determines if a system of 3D vectors is in equilibrium.
 * Equilibrium is achieved if the sum of all vectors results in a zero vector.
 *
 * @algorithm
 * The algorithm sums the components of all input vectors independently.
 * 1. Read the number of vectors, n.
 * 2. Initialize three variables (x, y, z) to store the sum of each vector
 *    component to zero.
 * 3. Loop n times, reading each vector's three components.
 * 4. Add each component to its respective sum variable.
 * 5. After the loop, check if all three sum variables are equal to zero.
 * 6. If x, y, and z are all zero, the system is in equilibrium, print "YES".
 * 7. Otherwise, print "NO".
 *
 * @time_complexity O(N), where N is the number of vectors, as we iterate through each vector once.
 * @space_complexity O(1), as the memory usage is constant regardless of the number of vectors.
 */
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  // Initialize sums of vector components for each axis.
  // x, y, and z will store the components of the resultant vector.
  int x = 0, y = 0, z = 0;
  int a, b, c;

  // Block Logic: Iterate through all N vectors, accumulating their components.
  // Invariant: At the end of each iteration, x, y, and z hold the sum of
  // components for all vectors processed so far.
  while (n--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  // Block Logic: Check if the resultant vector is the zero vector.
  // This condition is met if the sum of components along each axis is zero,
  // indicating that the system is in equilibrium.
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}
