/**
 * @file 231A.CPP
 * @brief A verbose and memory-inefficient solution to the "Team" problem.
 *
 * This program counts the number of problems a team of three will solve, where
 * a problem is solved if at least two members are confident.
 *
 * @b WARNING: This implementation is highly inefficient. It unnecessarily stores
 *    all inputs in large arrays before processing them, consuming O(N) space where
 *    O(1) is sufficient. The logic for checking the condition is also overly
 *    verbose compared to a simple summation.
 *
 * @b Algorithm: The program reads all votes for all problems into three separate
 *    arrays. It then iterates through the problems, using a long chain of if-else
 *    statements to check every possible combination of two or three members being
 *    confident.
 * @b Time_Complexity: O(N), where N is the number of problems.
 * @b Space_Complexity: O(N), due to storing all votes in arrays.
 */
#include <cmath>
#include <iostream>
using namespace std;
int main() {
  // Unnecessarily large arrays to store all votes.
  int num1[1000];
  int num2[1000];
  int num3[1000];
  int counter = 0;
  int y; // The number of problems.
  int i;

  cin >> y;

  for (i = 0; i < y; i++) {
    // Read all inputs first.
    cin >> num1[i] >> num2[i] >> num3[i];

    /**
     * @brief A verbose chain of conditions to check if at least two members agree.
     * A much simpler approach is: `if (num1[i] + num2[i] + num3[i] >= 2)`.
     */
    if (num1[i] == num2[i] && num1[i] == num3[i] && num1[i] == 1) // All three agree.
      counter++;
    else if (num1[i] == num2[i] && num1[i] == 1) // Members 1 and 2 agree.
      counter++;
    else if (num2[i] == num3[i] && num2[i] == 1) // Members 2 and 3 agree.
      counter++;
    else if (num1[i] == num3[i] && num1[i] == 1) // Members 1 and 3 agree.
      counter++;
  }
  
  cout << counter << endl;
  return 0;
}