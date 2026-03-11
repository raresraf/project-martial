/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D force vectors is in equilibrium.
 *
 * This program reads a set of 3D vectors and determines if their sum is the
 * zero vector (i.e., if the net force is zero). This is a classic physics
 * problem of checking for static equilibrium. The program reads the number
 * of vectors, then iteratively reads each vector's components and sums them up.
 * If the final sum of all x, y, and z components is zero, the system is in
 * equilibrium ("YES"), otherwise it is not ("NO").
 *
 * @algorithm Vector component summation. The program maintains three variables
 * to store the running sum of the x, y, and z components of all vectors.
 *
 * @time_complexity O(N), where N is the number of vectors, as the program
 * iterates through each vector once to read and sum its components.
 * @space_complexity O(1), as the program only uses a few variables to store
 * the running sums, regardless of the number of vectors.
 */
#include <iostream>

using namespace std;

int main() {
  int n; // The total number of force vectors.
  int a, b, c; // Variables to hold the components of each input vector.
  int r1, r2, r3; // Running sums for the x, y, and z components, respectively.
  r1 = r2 = r3 = 0;

  // Read the number of vectors.
  cin >> n;
  
  // The first vector's components are read and used to initialize the sums.
  // The loop will then run n-1 times for the remaining vectors.
  // This is a slightly unconventional way to structure the loop.
  n--; 
  cin >> r1 >> r2 >> r3;

  /**
   * @brief Loop to read the remaining vectors and accumulate their components.
   * @invariant At the beginning of each iteration, r1, r2, and r3 hold the
   * sum of components of the vectors processed so far.
   */
  while (n--) {
    // Read the components of the next vector.
    cin >> a >> b >> c;

    // Add the components to their respective running sums.
    r1 += a;
    r2 += b;
    r3 += c;
  }

  /**
   * @brief Check for equilibrium.
   * @pre r1, r2, and r3 contain the total sum of components for all vectors.
   * If the sum of all x, y, and z components is zero, the net force is zero.
   */
  if (!r1 && !r2 && !r3)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
