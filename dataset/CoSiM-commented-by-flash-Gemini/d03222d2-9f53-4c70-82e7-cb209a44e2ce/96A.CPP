/**
 * @file 96A.CPP
 * @brief This program solves the "Football" problem. It determines if a situation
 * in a football game is "dangerous", which is defined as having 7 or more
 * consecutive players of the same team. Teams are represented by '0's and '1's in an input string.
 * Algorithm: Linear scan through the input string, counting consecutive identical characters.
 * The count resets when a different character is encountered. The program terminates early
 * if a sequence of 7 or more is found.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for storing the input string.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
using namespace std;

int main() {
  string input; // Stores the input string representing the sequence of players (e.g., "001111110").
  // Reads the input string from standard input.
  cin >> input;

  int len = input.length(); // Stores the length of the input string.
  int counter = 1;          // Counts consecutive identical characters. Initializes to 1 for the first character.
  // Stores the integer value of the first character (0 or 1). ASCII '0' is 48, so subtracting 48 converts to int.
  int last = (int)input[0] - 48;
  
  /**
   * Block Logic: Iterates through the input string starting from the second character
   * to count consecutive identical digits. It uses 'last' to remember the previous digit
   * and 'counter' to store the current consecutive count. If the current digit is different
   * from 'last', 'counter' is reset to 1, and 'last' is updated. The loop breaks early
   * if 'counter' reaches 7.
   * Invariant: 'counter' holds the current length of the consecutive sequence of 'last' digit.
   */
  for (int i = 1; i < len; i++) {
    // If the current character is '0'.
    if ((int)input[i] - 48 == 0) {
      if (last == 0)
        counter++; // If the previous was also '0', increment counter.
      else {
        counter = 1; // If previous was '1', reset counter to 1 for new '0' sequence.
        last = 0;    // Update 'last' to '0'.
      }
    }
    // If the current character is '1'.
    if ((int)input[i] - 48 == 1) {
      if (last == 1)
        counter++; // If the previous was also '1', increment counter.
      else {
        counter = 1; // If previous was '0', reset counter to 1 for new '1' sequence.
        last = 1;    // Update 'last' to '1'.
      }
    }
    // If 'counter' reaches 7 at any point, a dangerous situation is found, so break early.
    if (counter == 7)
      break;
  }

  /**
   * Conditional Logic: After the loop, determines if a "dangerous" situation was found.
   * If 'counter' is 7, it means a sequence of 7 identical consecutive characters was found.
   * Otherwise, no such sequence was found.
   */
  if (counter == 7)
    cout << "YES"; // Prints "YES" if a dangerous situation is found.
  else
    cout << "NO";  // Prints "NO" otherwise.

  // Standard program exit.
  return 0;
}
