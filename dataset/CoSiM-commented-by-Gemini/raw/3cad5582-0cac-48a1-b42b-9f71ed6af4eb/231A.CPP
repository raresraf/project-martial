/**
 * @file 231A.CPP
 * @brief A robust solution to the "Team" problem (Codeforces 231A) with input validation.
 *
 * @algorithm
 * The problem is to count how many problems a three-person team will implement.
 * A problem is chosen if at least two members are confident in the solution.
 *
 * This implementation follows a direct simulation approach with added safety checks:
 * 1. Read the total number of problems, `n`.
 * 2. Validate that `n` is within the problem's specified constraints (1 to 1000).
 * 3. Initialize a counter `number` for solvable problems.
 * 4. Loop `n` times using a `while` loop.
 * 5. In each iteration, read the three members' confidence values.
 * 6. Validate that each confidence value is either 0 or 1.
 * 7. If the sum of the three values is 2 or greater, increment the counter.
 * 8. After the loop, print the final count.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of problems. Each problem is
 *   processed in constant time.
 * - Space Complexity: O(1), as memory usage is constant.
 */
#include <iostream>

using namespace std;

int main() {
  // n: total problems. pet, vas, ton: confidence of team members. number: result counter.
  int n, pet, vas, ton, number(0);

  cin >> n;

  /**
   * @brief Validates that the number of problems `n` is within the allowed
   * constraints [1, 1000] as specified by the problem statement.
   */
  if (!(1 <= n && n <= 1000)) {
    return 0; // Exit if `n` is out of bounds.
  } else {
    ; // This empty else block has no effect.
  }

  /**
   * @brief Main loop to process each problem. The loop continues `n` times.
   * Invariant: `number` holds the count of solvable problems processed so far.
   */
  while (n--) {
    cin >> pet >> vas >> ton;
    
    /**
     * @brief Validates that each team member's input is either 0 or 1.
     * This is a creative, though verbose, way to ensure input integrity.
     */
    for (int x : {pet, vas, ton}) {
      switch (x) {
      case 0:
      case 1:
        break; // Input is valid.
      default:
        return 0; // Exit if input is invalid.
      }
    }
    
    /**
     * @brief The core logic: if the sum of confidences is 2 or more, the team
     * will solve the problem.
     */
    if (pet + vas + ton >= 2) {
      number += 1;
    }
  }
  cout << number << endl;
  return 0;
}