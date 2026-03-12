/**
 * @605c39f3-0e75-4d2f-b34d-5b8cc52ab0f7/898B.CPP
 * @brief This program solves a linear Diophantine equation of the form `x*a + y*b = n`
 * for non-negative integer solutions `x` and `y`. It searches for a pair of non-negative
 * coefficients that satisfy the equation for given `n`, `a`, and `b`.
 *
 * Algorithm: Iterative search (brute force) for non-negative integer solutions.
 * Time Complexity: O(N/A) in the worst case, where N is the target sum and A is the coefficient 'a'.
 *                  The loop runs at most `n/a + 1` times.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // For standard input/output operations (std::cin, std::cout).
#include <vector>   // Included but not explicitly used in this snippet.
#include <cstdio>   // For printf and puts.

int main() {
  long n, a, b; // Declares 'n' (target sum), 'a' (first coefficient), 'b' (second coefficient).
  std::cin >> n >> a >> b; // Reads n, a, and b from standard input.
  long x(-1), y(-1); // Initializes 'x' and 'y' to -1, acting as sentinel values to indicate no solution found.
  
  /**
   * Block Logic: Iterates through possible non-negative integer values for 'x' (represented by 'p').
   * The loop continues as long as `p*a` does not exceed the target sum 'n'.
   * Invariant: If a solution is found, 'x' and 'y' are updated to non-negative values, and the loop breaks.
   */
  for (long p = 0; p * a <= n; p++) { // 'p' represents the potential value for 'x'.
    // Inline: Checks if the remaining amount (n - p*a) is perfectly divisible by 'b'.
    // If it is, then 'y' can be a non-negative integer.
    if ((n - p * a) % b == 0) { 
      x = p; // Assign the current 'p' to 'x'.
      y = (n - p * a) / b; // Calculate the corresponding 'y'.
      break; // A non-negative solution is found, so exit the loop.
    }
  }
  
  /**
   * Block Logic: Prints the result based on whether a non-negative solution was found.
   * Pre-condition: 'x' and 'y' contain the solution or remain -1.
   * Invariant: Prints "YES" and the solution if found, otherwise "NO".
   */
  if (x >= 0) { // If 'x' is not -1, a solution was found.
    printf("YES\n%ld %ld\n", x, y); // Prints "YES" followed by the values of x and y.
  } else { // No non-negative solution was found.
    puts("NO"); // Prints "NO".
  }

  return 0; // Indicates successful program termination.
}
