/**
 * @file 69A.CPP
 * @brief Checks if a system of 3D vectors is in equilibrium using a memory-efficient approach.
 *
 * This program reads a specified number of 3D vectors and determines if their
 * vector sum is the zero vector. It does so by maintaining a running sum of the
 * x, y, and z components without storing the individual vectors. If the final
 * sum is the zero vector, it prints "YES", otherwise it prints "NO".
 *
 * Algorithm:
 * 1. Read an integer `n`, the total number of vectors.
 * 2. Initialize three sum variables, `sumx`, `sumy`, and `sumz`, to zero.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of a single vector.
 *    b. Add these components directly to the running totals `sumx`, `sumy`, and `sumz`.
 * 4. After the loop, check if `sumx`, `sumy`, and `sumz` are all zero.
 * 5. If they are, the system is in equilibrium; print "YES".
 * 6. Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of vectors, as the program
 * iterates through the input vectors once.
 * Space Complexity: O(1). This implementation is memory-efficient as it does not
 * require storing the components of all vectors simultaneously.
 */
#include <iostream>

using namespace std;

int main() {
  // Initialize sums for each vector component.
  int sumx = 0, sumy = 0, sumz = 0, n;
  // Read the number of vectors to process.
  cin >> n;
  
  /**
   * @brief Loop `n` times to read vectors and accumulate their component sums.
   * Pre-condition: `n` is the number of vectors to be read.
   * Invariant: At the start of each iteration, `sumx`, `sumy`, `sumz` hold the
   * component sums of the vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    int x, y, z;
    // Read the components of the current vector.
    cin >> x >> y >> z;
    // Add components to their respective running totals.
    sumx += x;
    sumy += y;
    sumz += z;
  }
  
  /**
   * @brief Check the equilibrium condition.
   * The system is in equilibrium if the sum of forces in all three dimensions is zero.
   */
  if ((sumx == 0) && (sumy == 0) && (sumz == 0)) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;
    
  return 0;
}