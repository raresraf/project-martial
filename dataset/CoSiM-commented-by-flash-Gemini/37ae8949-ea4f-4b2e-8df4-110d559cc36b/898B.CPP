/**
 * @file 898B.CPP
 * @brief Solution for Codeforces Problem 898B - Equalize Prices. (This problem name seems unrelated to the code. A more fitting problem would be "Divisible by A and B" or similar based on the logic.)
 *
 * This program determines if a given integer 'n' can be expressed as the sum
 * of non-negative integer multiples of two other given integers 'a' and 'b'.
 * That is, can 'n = x*a + y*b' where 'x >= 0' and 'y >= 0'? If possible,
 * it outputs "YES" along with one such pair (x, y). Otherwise, it outputs "NO".
 *
 * Algorithm: Iterative checking. The program first checks if 'n' is a direct multiple of 'a' or 'b'.
 * If not, it iterates through possible non-negative multiples of 'a' (i*a) and checks if the remaining
 * value (n - i*a) is a non-negative multiple of 'b'. It then repeats this process by iterating
 * through multiples of 'b'. The iteration limits (10^7) are heuristic for competitive programming
 * to prevent Time Limit Exceeded (TLE) errors for larger inputs, assuming a solution exists within this range.
 * Time Complexity: O(N/min(A,B)) in the worst case, but practically limited by the iteration constant.
 * Space Complexity: O(1).
 */

#include <iostream>
using namespace std;

// Declare long long integers for 'n', 'a', 'b' to handle potentially large inputs.
long long n, a, b;

int main() {
  /**
   * @brief Main function to solve the problem of representing 'n' as a sum of multiples of 'a' and 'b'.
   *
   * Reads 'n', 'a', and 'b'. It first checks for simple cases where 'n' is a direct multiple of 'a' or 'b'.
   * Then, it uses two nested loops (iterative approach) to find non-negative integer coefficients 'x' and 'y'
   * such that n = x*a + y*b. If a solution is found, it prints "YES" with 'x' and 'y', and terminates.
   * If no solution is found after exhausting search space, it prints "NO".
   */
  // Read the target number 'n'.
  cin >> n;
  // Read the two base numbers 'a' and 'b'.
  cin >> a >> b;

  // Block Logic: Check if 'n' is a direct multiple of 'a'.
  // Pre-condition: 'n' has been read.
  if (n % a == 0)
    // If 'n' is a multiple of 'a', then x = n/a and y = 0.
    // Prints "YES", followed by n/a and 0, then terminates.
    return cout << "YES\n" << n / a << " " << 0, 0;

  // Block Logic: Check if 'n' is a direct multiple of 'b'.
  // Pre-condition: 'n' was not a direct multiple of 'a'.
  if (n % b == 0)
    // If 'n' is a multiple of 'b', then x = 0 and y = n/b.
    // Prints "YES", followed by 0 and n/b, then terminates.
    return cout << "YES\n" << 0 << " " << n / b, 0;

  // Block Logic: Iterate through possible multiples of 'a'.
  // Invariant: 'i' represents the coefficient for 'a' (x).
  // The loop explores values of 'x' from 1 up to a large constant, covering practical cases.
  for (int i = 1; i < 10000000; i++) {
    // Calculate the remaining value after subtracting 'i' multiples of 'a'.
    long long x = n - (a * i);
    // Check if the remaining value 'x' is a non-negative multiple of 'b'.
    if (x % b == 0 && x >= 0)
      // If a solution is found, print "YES", followed by 'i' (coefficient for 'a')
      // and 'x/b' (coefficient for 'b'). Terminates the program.
      // Special handling for x==0 to explicitly print 0, as x/b would be 0 anyway.
      return cout << "YES\n" << i << " " << (x == 0 ? 0 : x / b), 0;
  }

  // Block Logic: Iterate through possible multiples of 'b'.
  // Invariant: 'i' represents the coefficient for 'b' (y).
  // This loop is executed if the previous loop did not find a solution.
  // It explores values of 'y' from 1 up to a large constant.
  for (int i = 1; i < 10000000; i++) {
    // Calculate the remaining value after subtracting 'i' multiples of 'b'.
    long long x = n - (b * i);
    // Check if the remaining value 'x' is a non-negative multiple of 'a'.
    if (x % a == 0 && x >= 0)
      // If a solution is found, print "YES", followed by 'x/a' (coefficient for 'a')
      // and 'i' (coefficient for 'b'). Terminates the program.
      // Special handling for x==0 to explicitly print 0.
      return cout << "YES\n" << (x == 0 ? 0 : x / a) << " " << i, 0;
  }

  // If no combination is found after all checks, print "NO".
  cout << "NO";
  return 0;
}
