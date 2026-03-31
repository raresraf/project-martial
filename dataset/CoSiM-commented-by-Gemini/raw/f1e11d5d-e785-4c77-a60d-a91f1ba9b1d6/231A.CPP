/**
 * @file 231A.CPP
 * @brief Counts solvable problems based on team member confidence.
 *
 * This program solves the "Team" problem by determining how many problems a three-person
 * team will attempt. It reads an integer `n`, the number of problems. For each
 * problem, it reads the confidence of each of the three members (as 1 for confident,
 * 0 for not). If two or more members are confident, the team attempts the problem.
 * The program outputs the total count of problems they will attempt.
 *
 * @author f1e11d5d-e785-4c77-a60d-a91f1ba9b1d6
 *
 * @algorithm Nested Loop Counting
 * @complexity
 * Time: O(N), where N is the number of problems. The inner loop runs a constant
 *       3 times, so the complexity is 3*N, which simplifies to O(N).
 * Space: O(1), as memory usage is constant.
 */
#include <iostream>

using namespace std;

int main() {
  // `n`: number of problems.
  // `opinion`: stores the confidence of a single team member (1 or 0).
  // `sure`: counts confident members for the current problem.
  int n, opinion, sure;
  // `problems`: counts the total number of problems to be solved.
  int problems = 0;

  cin >> n;

  /**
   * @brief Outer loop iterates through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    // Reset the `sure` counter for each new problem.
    sure = 0;
    /**
     * @brief Inner loop reads the opinions of the 3 team members.
     */
    for (int j = 0; j < 3; j++) {
      cin >> opinion;
      // If a member is confident (opinion == 1), increment the `sure` counter.
      if (opinion)
        sure++;
    }
    /**
     * @brief Check if the problem will be attempted.
     * The condition is met if more than one member is confident.
     */
    if (sure > 1)
      problems++;
  }

  // Output the final count of solvable problems.
  cout << problems << endl;

  return 0;
}