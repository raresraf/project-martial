/**
 * @file 231A.CPP
 * @brief A clear and correct solution to the "Team" problem (Codeforces 231A).
 *
 * @algorithm
 * The problem is to count the number of problems a three-person team will solve,
 * where a problem is solved if at least two members are confident.
 *
 * This implementation uses a straightforward counting method:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a primary counter `c` for the final answer.
 * 3. Loop `n` times to process each problem.
 * 4. For each problem, initialize a temporary local counter `count`.
 * 5. Read the three confidence values (0 or 1). For each `1` read, increment
 *    the local `count`.
 * 6. After reading the three values, if `count` is 2 or greater, it means the
 *    problem will be solved, so increment the primary counter `c`.
 * 7. After the main loop, print the value of `c`.
 *
 * @complexity
 * - Time Complexity: O(N), for a single pass through the N problems.
 * - Space Complexity: O(1), as storage is constant regardless of N.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @note The use of `main()` without an explicit `int` return type is
 * non-standard in C++. `int main()` is the standard signature.
 */
main() {
  int n; // The total number of problems.
  cin >> n;
  int c = 0; // The final count of problems that will be implemented.
  /**
   * @brief Main loop to process each of the `n` problems.
   */
  while (n--) {
    int a[3];      // Array to store the three opinions for the current problem.
    int count = 0; // Local counter for confident members for this problem.
    /**
     * @brief Inner loop to read three opinions and count the confident ones.
     */
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1)
        count++;
    }
    // If two or more members are confident, increment the final counter.
    if (count >= 2)
      c++;
  }
  cout << c;
}
