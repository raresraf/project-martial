#include <cstdio>
using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem using nested loops.
 *
 * Problem Statement:
 * A team of three members will only solve a problem if at least two of them
 * are confident. Given the confidence status (1 for confident, 0 otherwise)
 * for `n` problems, this program calculates the total number of problems the
 * team will solve.
 *
 * Algorithm:
 * This implementation uses nested loops to process the input.
 * 1. Read the total number of problems, `n`.
 * 2. An outer loop iterates `n` times, once for each problem.
 * 3. Inside the outer loop, a `count` variable is reset to zero.
 * 4. An inner loop runs 3 times to read the confidence level of each of the
 *    three members for the current problem.
 * 5. If a member is confident (input is 1), the `count` is incremented.
 * 6. After the inner loop finishes, if `count` is 2 or more, a `solve`
 *    counter is incremented.
 * 7. Finally, the total value of `solve` is printed.
 *
 * @note The code uses C-style I/O (`scanf`, `printf`) and custom preprocessor
 * macros for loops, which are common in competitive programming for speed.
 *
 * Time Complexity: O(N), as the inner loop runs a fixed 3 times for each of
 * the N problems.
 * Space Complexity: O(1).
 */

// Custom macros for writing for-loops concisely.
#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i--)

int main() {
  int n;     // Total number of problems.
  int temp;  // Temporary storage for a member's confidence level.
  int count; // Counts confident members for a single problem.
  int solve = 0; // Counts total problems to be solved.

  scanf("%d", &n);

  // Outer loop: iterate through each problem.
  rep(i, 0, n) {
    // Reset the confidence counter for the current problem.
    count = 0;
    // Inner loop: iterate through the three members' opinions.
    rep(j, 0, 3) {
      scanf("%d", &temp);
      // If the member is confident (input is 1), increment the count.
      if (temp)
        count++;
    }
    // Pre-condition: Check if at least two members are confident.
    if (count >= 2)
      solve++;
  }

  printf("%d\n", solve);
}