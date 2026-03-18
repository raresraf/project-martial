/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem: "Football".
 * Determines if a "dangerous" situation occurs in a football match,
 * defined as seven consecutive players of the same team.
 * Algorithm: Linear scan to count consecutive identical characters.
 * Time Complexity: O(L) where L is the length of the input string (situation).
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Football problem.
 * Reads a string representing the football situation and checks for seven consecutive
 * identical characters to determine if it's "dangerous".
 */
int main() {
  string situation;
  cin >> situation;
  /**
   * Block Logic: Initial check for string length.
   * If the string is shorter than 7, a "dangerous" situation is impossible.
   * Pre-condition: 'situation' contains the input string.
   */
  if (situation.size() < 7) {
    cout << "NO";
  } else {
    int counter = 1;
    /**
     * Block Logic: Iterates through the string to count consecutive identical characters.
     * Pre-condition: 'situation' has a length of at least 7. 'counter' is initialized to 1.
     * Invariant: 'counter' tracks the current streak of identical characters.
     */
    for (int i = 1; i < situation.size(); i++) {
      /**
       * Block Logic: Increments counter if current character matches the previous one.
       */
      if (situation[i] == situation[i - 1]) {
        counter += 1;
      }
      /**
       * Block Logic: Resets counter if characters differ and the streak is not yet dangerous.
       * If the streak was less than 7, it's not "dangerous", so restart counting.
       */
      else if (counter < 7) {
        counter = 1;
      }
    }
    /**
     * Block Logic: Final determination after checking the entire string.
     * If 'counter' is 7 or more, a "dangerous" situation was found.
     */
    if (counter >= 7) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  }

  return 0;
}
