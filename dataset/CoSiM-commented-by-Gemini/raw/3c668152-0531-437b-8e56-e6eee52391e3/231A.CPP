/**
 * @file 231A.CPP
 * @brief A concise solution to the "Team" problem (Codeforces 231A).
 *
 * @algorithm
 * The problem is to count the number of programming contest problems that will be
 * implemented by a three-person team. A problem is implemented if at least two
 * members are confident about its solution.
 *
 * The algorithm implemented is a direct simulation:
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop `t` times to process each problem.
 * 4. For each problem, read the three members' confidence values (1 for confident,
 *    0 otherwise).
 * 5. Sum the three confidence values. If the sum is 2 or greater, it signifies
 *    that at least two members agree.
 * 6. If the condition is met, increment the problem counter.
 * 7. After iterating through all problems, print the final count.
 *
 * @complexity
 * - Time Complexity: O(T), where T is the number of problems, because each
 *   problem is processed in a single pass with constant time operations.
 * - Space Complexity: O(1), as the memory usage is constant and does not
 *   depend on the number of problems.
 */
#include <iostream>
using namespace std;
int main() {
  int t; // Represents the total number of problems.
  cin >> t;
  int count = 0; // Counter for problems that will be implemented.
  /**
   * @brief Loop to iterate through each problem.
   * Pre-condition: `t` holds the number of problems to process.
   * Invariant: At the end of each iteration `i`, `count` holds the number of
   * solvable problems among the first `i+1` problems.
   */
  for (int i = 0; i < t; i++) {
    int a, b, c; // Confidence levels of the three team members.
    cin >> a >> b >> c;
    /**
     * @brief Checks if the sum of confidence levels is 2 or more.
     * This is a clean and efficient way to determine if at least two members
     * are confident about the solution.
     */
    if (a + b + c >= 2)
      count++;
  }
  cout << count << endl;
}
