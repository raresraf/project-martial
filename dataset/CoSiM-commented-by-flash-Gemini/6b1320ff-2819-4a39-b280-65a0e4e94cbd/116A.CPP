/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, passengers exit (`a`)
 * and then enter (`b`). The program efficiently tracks the current number of passengers (`c`) and the
 * maximum number of passengers observed (`max`) throughout the journey. It uses 1-indexed arrays for input
 * and `long int` for maximum passenger count.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(N)$ to store exit and entry counts, or effectively $O(1)$ if 'c' and 'max' are considered.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, passengers exit (`a`)
 * and then enter (`b`). The program efficiently tracks the current number of passengers (`c`) and the
 * maximum number of passengers observed (`max`) throughout the journey. It uses 1-indexed arrays for input
 * and `long int` for maximum passenger count.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(N)$ to store exit and entry counts, or effectively $O(1)$ if 'c' and 'max' are considered.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, algorithm).

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the tram simulation logic.
   */
  int n,     // @var n: The total number of tram stops.
      i,     // @var i: Loop counter.
      a[1002], // @var a: Array to store the number of passengers exiting at each stop (1-indexed).
      b[1002], // @var b: Array to store the number of passengers entering at each stop (1-indexed).
      c;     // @var c: Stores the current number of passengers in the tram.
  long int max; // @var max: Stores the maximum number of passengers encountered in the tram, using `long int` for safety.

  scanf("%d", &n); // Reads the total number of tram stops.

  /**
   * @brief Reads the number of exiting and entering passengers for each stop.
   * @block_logic Iterates `n` times to populate arrays `a` and `b`.
   * @invariant For each `i` from 1 to `n`, `a[i]` and `b[i]` store the respective passenger counts.
   * @note Arrays `a` and `b` are used 1-indexed.
   */
  for (i = 1; i <= n; i++) {
    scanf("%d %d", &a[i], &b[i]); // Reads exit and entry counts for stop `i`.
  }

  /**
   * @brief Initializes the passenger count after the first stop (Stop 1).
   * @details The initial passenger count `c` is set to the sum of `a[1]` and `b[1]`.
   *          Typically, this would represent `entries - exits` for an empty tram.
   *          Given the variable names, `a[1]` is exits and `b[1]` is entries.
   *          The logic here might assume some initial passengers or a different interpretation.
   */
  c = a[1] + b[1]; // @var c: Passengers after the first stop.
  max = c;         // Initialize max with passengers after the first stop.

  /**
   * @brief Simulates the tram journey through each subsequent stop, updating passenger counts and tracking the maximum.
   * @block_logic Iterates from the second stop (`i=1`) up to the last stop (`n-1`).
   * @invariant `c` reflects the current passenger count, and `max` holds the peak passenger count.
   * @note The loop reads `a[i+1]` and `b[i+1]`, correctly using the next stop's data.
   */
  for (i = 1; i <= n - 1; i++) {
    // Calculates the current number of passengers: (passengers before this stop) - (net change at next stop).
    // `(a[i+1] - b[i+1])` represents the net decrease in passengers at the next stop.
    // So `c - (a[i+1] - b[i+1])` is `c - exits + entries`.
    c = c - (a[i + 1] - b[i + 1]); // Updates the current number of passengers.
    if (c > max) { // Checks if the current passenger count is a new maximum.
      max = c;     // Updates `max` if the current `c` is higher.
    }
  }

  // Functional Utility: Prints the minimum capacity required (maximum number of passengers observed).
  cout << max << endl;
  return 0; // Indicates successful execution.
}
