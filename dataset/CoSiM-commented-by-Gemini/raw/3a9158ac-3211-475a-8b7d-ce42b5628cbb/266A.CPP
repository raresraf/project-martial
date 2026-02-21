/**
 * @file 266A.CPP
 * @brief Counts adjacent identical stones that need to be removed.
 *
 * This program calculates the minimum number of stones to take from a row
 * to ensure no two adjacent stones share the same color. It achieves this by
 * counting every instance where a stone is followed immediately by another
 * of the same color. The outer `while` loop suggests it was designed to handle
 * multiple test cases.
 *
 * Algorithm:
 * 1. An outer loop reads the number of stones `cs`, continuing as long as
 *    there is input.
 * 2. Inside the loop, a counter `count` is reset to 0.
 * 3. A string `inp` representing the stone colors is read.
 * 4. The program iterates through the string `inp`.
 * 5. It uses a variable `last_ch` to keep track of the previous stone's color.
 * 6. For the very first stone (`i == 0`), it initializes `last_ch` and skips
 *    to the next iteration.
 * 7. For every subsequent stone, it compares its color (`ch`) with `last_ch`.
 * 8. If they are the same, `count` is incremented.
 * 9. `last_ch` is updated to the current stone's color for the next comparison.
 * 10. After the inner loop, the total `count` is printed.
 *
 * Time Complexity: O(T * N), where T is the number of test cases and N is the
 * number of stones in each case. For a single case, it is O(N).
 * Space Complexity: O(N) to store the input string `inp`.
 */
#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, count = 0;
  string inp;
  char ch, last_ch;

  // Outer loop to handle multiple test cases.
  while (cin >> cs) {
    count = 0;
    cin >> inp;

    /**
     * @brief Loop through the string to compare adjacent characters.
     * Pre-condition: `inp` holds the string of stone colors.
     */
    for (int i = 0; i < inp.size(); i++) {
      // Handle the first character separately to initialize `last_ch`.
      if (i == 0) {
        last_ch = inp[i];
        continue;
      }
      ch = inp[i];
      // If the current character is the same as the last one, increment count.
      if (ch == last_ch) {
        count++;
      }
      // Update `last_ch` for the next iteration.
      last_ch = ch;
    }
    cout << count << endl;
  }
}
