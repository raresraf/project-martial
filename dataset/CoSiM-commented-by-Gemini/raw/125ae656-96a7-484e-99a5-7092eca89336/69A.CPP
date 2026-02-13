/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @author competitive-programmer
 *
 * @details This program determines if a system is in equilibrium by checking if
 * the vector sum of a given set of 3D force vectors is zero.
 *
 * Algorithm (Two-Pass Approach):
 * 1. Read the number of force vectors, `lines`.
 * 2. Store all the x, y, and z components of the vectors in three separate arrays.
 * 3. In a second pass, iterate through the arrays and compute the sum of all x,
 *    y, and z components respectively.
 * 4. A system is in equilibrium if and only if the net force along each axis is zero.
 *    Check if the sums of all three components are zero.
 * 5. Print "YES" if the system is in equilibrium, otherwise print "NO".
 *
 * @note This implementation uses a two-pass approach and stores all input values,
 * which is less memory-efficient than a single-pass approach that accumulates the
 * sums directly without storing the individual vectors.
 *
 * Time Complexity: O(n), where n is the number of vectors.
 * Space Complexity: O(n) to store the components of the n vectors.
 */
#include <bits/stdc++.h>
#include <iostream>

int main() {
  int lines;
  std::cin >> lines;
  
  // Note: The use of Variable Length Arrays (VLAs) like x[lines] is a C99 feature
  // and not part of standard C++. A std::vector would be the standard C++ approach.
  int x[lines], y[lines], z[lines];
  int sumx = 0, sumy = 0, sumz = 0;

  /**
   * Block Logic (First Pass): Read all vector components and store them in arrays.
   */
  for (int i = 0; i < lines; i++) {
    std::cin >> x[i] >> y[i] >> z[i];
  }

  /**
   * Block Logic (Second Pass): Calculate the sum of forces for each axis.
   */
  for (int i = 0; i < lines; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  /**
   * Decision Logic: The system is in equilibrium if the net force along each
   * axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0) {
    std::cout << "YES";
  } else {
    std::cout << "NO";
  }

  return 0;
}
