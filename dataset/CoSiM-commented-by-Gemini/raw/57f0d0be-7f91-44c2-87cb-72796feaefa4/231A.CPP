/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three will solve.
 * The team will solve a problem only if at least two members are sure about the solution.
 *
 * @algorithm
 * The solution reads the total number of problems, `n`. It then loops `n` times.
 * In each iteration, it reads the opinions of the three members (Petya, Vasya, Tonya)
 * as binary values (1 for sure, 0 for not sure). It sums these values to get a
 * "sureness" count for the current problem. If this count is 2 or greater, it
 * increments a counter for solvable problems. Finally, it prints the total count.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, as the solution involves
 * a single loop that iterates N times with constant-time operations inside.
 * Space Complexity: O(1), as only a few integer variables are used regardless of the input size.
 */
#include <iostream>

using namespace std;

int main() {

  /**
   * @variable n: The total number of problems in the contest.
   */
  int n;
  /**
   * @variable p, v, t: Represent the confidence of Petya, Vasya, and Tonya, respectively.
   * 1 means sure, 0 means not sure.
   */
  int p, v, t;
  p = v = t = 0;
  /**
   * @variable ans: Counter for the number of problems the team will implement.
   */
  int ans = 0;
  // Read the total number of problems.
  cin >> n;

  /**
   * @block
   * @brief This loop iterates through each of the `n` problems.
   * @invariant `ans` holds the count of problems solvable by the team so far.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @variable count: Stores the number of team members sure about the current problem.
     */
    int count = 0;
    // Read the opinions of the three members for the current problem.
    cin >> p >> v >> t;
    // Pre-condition: Check if Petya is sure.
    if (p == 1)
      count++;
    // Pre-condition: Check if Vasya is sure.
    if (v == 1)
      count++;
    // Pre-condition: Check if Tonya is sure.
    if (t == 1)
      count++;

    // Pre-condition: Check if the number of sure members is at least two.
    if (count > 1)
      ans++; // If so, increment the answer counter.
  }
  // Output the total number of problems the team will solve.
  cout << ans << endl;
  return 0;
}
