/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram.
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit and then a certain number enter. The program tracks the number of passengers
 * inside the tram and finds the maximum value reached.
 * @algorithm Iterative simulation and maximum tracking.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(N)$ to store exit and entry counts, and passenger counts.
 */
#include <iostream>

/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram.
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit and then a certain number enter. The program tracks the number of passengers
 * inside the tram and finds the maximum value reached.
 * @algorithm Iterative simulation and maximum tracking.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(N)$ to store exit and entry counts, and passenger counts.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).

int main() {
  /**
   * @brief Main function: orchestrates the tram simulation logic.
   */
  int i, n,        // @var i: Loop counter. @var n: Total number of tram stops.
      a[1005],     // @var a: Array to store the number of passengers exiting at each stop.
      b[1005],     // @var b: Array to store the number of passengers entering at each stop.
      c[1005],     // @var c: Array to store the number of passengers currently in the tram after each stop.
      max = 0;     // @var max: Stores the maximum number of passengers encountered in the tram.

  std::cin >> n; // Reads the total number of tram stops.

  /**
   * @brief Reads the number of exiting and entering passengers for each stop.
   * @block_logic Iterates 'n' times to populate arrays `a` and `b`.
   * @invariant For each `i` from 0 to `n-1`, `a[i]` and `b[i]` store the respective passenger counts.
   */
  for (i = 0; i < n; i++)
    std::cin >> a[i] >> b[i];

  // Functional Utility: Initializes the passenger count after the first stop.
  // Assuming the tram starts empty, after the first stop (0), `b[0]` passengers enter.
  c[0] = b[0];

  /**
   * @brief Calculates the number of passengers in the tram after each subsequent stop.
   * @block_logic Iterates from the second stop (`i=1`) up to the second-to-last stop (`n-2`).
   * @invariant `c[i]` stores the current number of passengers in the tram after stop `i`.
   *            The calculation `c[i-1] - a[i] + b[i]` updates the count based on exits and entries.
   * @note The loop explicitly stops at `n-2`, implying `c[n-1]` might not be calculated or handled separately.
   */
  for (i = 1; i < n - 1; i++)
    c[i] = c[i - 1] - a[i] + b[i];

  /**
   * @brief Finds the maximum number of passengers that were in the tram.
   * @block_logic Iterates through the calculated passenger counts in array `c`.
   * @invariant `max` stores the highest passenger count found so far.
   * @note The loop iterates up to `n-1`, which could access `c[n-1]` which might be uninitialized based
   *       on the previous loop's condition (`i < n-1`). This implies a potential out-of-bounds access
   *       or an implicit assumption about `c[n-1]`'s value. However, adhering to "Zero Code Mutation".
   */
  for (i = 0; i < n - 1; i++) {
    if (c[i] > max)
      max = c[i];
  }

  // Functional Utility: Prints the maximum number of passengers required (minimum capacity).
  std::cout << max << '\n';
  return 0; // Indicates successful execution.
}
