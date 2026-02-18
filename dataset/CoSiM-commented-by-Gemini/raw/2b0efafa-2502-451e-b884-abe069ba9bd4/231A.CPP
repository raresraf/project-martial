/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (231A) from Codeforces.
 * @details The program counts the number of problems a three-person team will implement.
 * A problem is implemented if at least two members are sure about the solution.
 * This version uses C-style input/output.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution iterates through each problem and checks if the number of
 * confident members meets the threshold of two.
 * 1. Read the total number of problems (n).
 * 2. Initialize a counter `ans` to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three confidence levels (0 or 1).
 * 5. If the sum of the levels is greater than 1 (i.e., 2 or 3), increment `ans`.
 * 6. After the loop, print the final count `ans`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <cstdio>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0;
  scanf("%d", &n);
  /**
   * @block
   * @brief Processes each of the 'n' problems to count how many will be solved.
   * @invariant 'ans' holds the count of problems solved so far.
   * @pre 'n' is the total number of problems.
   * @post 'ans' holds the total count where at least two members are confident.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    // A problem is implemented if the sum of confidences (0 or 1) is at least 2.
    if (a + b + c > 1)
      ans++;
  }
  printf("%d\n", ans);
  return 0;
}
