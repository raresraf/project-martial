/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces by checking for consecutive players.
 *
 * This program determines if a situation is "dangerous" by checking if a string
 * representing players ('0's and '1's) contains a sequence of 7 or more
 * identical characters.
 *
 * @algorithm The code performs a linear scan of the input string. It uses two
 * counters: `one` for consecutive '1's and `zero` for consecutive '0's. As it
 * iterates, it increments the appropriate counter and resets the other. If at any
 * point either counter reaches 7, a flag is set, and the loop terminates early.
 *
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(N) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  string str;
  int one = 0, zero = 0, flag = 0;
  cin >> str;
  /**
   * @block_logic Iterates through the string, updating counters for consecutive '1's and '0's.
   */
  for (int i = 0; i < str.length(); i++) {
    // If the character is '1', increment the 'one' counter and reset the 'zero' counter.
    if (str[i] == '1') {
      one++;
      zero = 0;
    } else {
      // Otherwise, it's '0', so increment 'zero' and reset 'one'.
      zero++;
      one = 0;
    }
    /**
     * @invariant After each character, check if a sequence of 7 has been formed.
     * If so, set the flag and break the loop as the condition has been met.
     */
    if (zero == 7 || one == 7) {
      flag = 1;
      break;
    }
  }
  // Based on the flag, print the result.
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
