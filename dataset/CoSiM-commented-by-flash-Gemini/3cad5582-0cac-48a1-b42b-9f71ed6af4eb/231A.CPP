/**
 * @file 231A.CPP
 * @brief Solution for competitive programming problem: "Team".
 *
 * @details This program determines the number of problems that can be implemented
 *          given 'n' problems and three participants' votes for each. A problem
 *          is considered implementable if at least two participants vote positively (1).
 *          Algorithm: Direct simulation and counting based on conditional logic.
 *          Time Complexity: O(N), where N is the number of problems, as each problem's
 *                           votes are processed in constant time.
 *          Domain: Competitive Programming.
 */
#include <iostream>

using namespace std;

int main() {
  // Declare variables: 'n' for the total number of problems,
  // 'pet', 'vas', 'ton' for votes of three participants, and 'number' to count implementable problems.
  int n, pet, vas, ton, number(0);

  // Read the total number of problems from standard input.
  cin >> n;

  // Block Logic: Validate the input 'n' to be within the allowed range [1, 1000].
  // Precondition: 'n' must be between 1 and 1000, inclusive.
  if (!(1 <= n && n <= 1000)) {
    return 0; // Terminate if 'n' is out of range.
  } else {
    // This empty else block has no functional impact but is part of the original code structure.
    ;
  }

  // Block Logic: Process each problem one by one.
  // Invariant: The loop runs 'n' times, once for each problem.
  while (n--) {
    // Read the votes of the three participants for the current problem.
    cin >> pet >> vas >> ton;
    
    // Block Logic: Validate each participant's vote to be either 0 or 1.
    // Precondition: Each vote (pet, vas, ton) must be 0 or 1.
    for (int x : {pet, vas, ton}) {
      switch (x) {
      case 0: // Valid vote (no).
      case 1: // Valid vote (yes).
        break;
      default: // Invalid vote.
        return 0; // Terminate if an invalid vote is encountered.
      }
    }
    // Block Logic: Check if the current problem is implementable.
    // Precondition: A problem is implementable if the sum of positive votes is 2 or more.
    if (pet + vas + ton >= 2) {
      number += 1; // Increment the count of implementable problems.
    }
  }
  // Output the total number of implementable problems.
  cout << number << endl;
  
  return 0; // Indicate successful program termination.
}