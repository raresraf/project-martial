/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - Stones on the Table.
 *
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones are of the same color. It achieves this
 * by counting adjacent identical stones. It is designed to handle multiple
 * test cases.
 *
 * Algorithm: Linear scan with state tracking. The program iterates through
 * the sequence of stones, keeping track of the last processed stone's color.
 * If the current stone's color matches the last stone's color, a counter
 * for adjacent identical stones is incremented.
 * Time Complexity: O(N), where N is the length of the string of stones.
 * Space Complexity: O(N) for storing the input string.
 */

#include <algorithm> // Not explicitly used in the main logic (sorting not needed here).
#include <cstdio>    // Not explicitly used in the main logic (uses iostream).
#include <iostream>  // Required for cin, cout.
#include <string>    // Required for std::string.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Stones on the Table problem logic.
   *
   * Reads an integer 'cs' (likely representing a dummy value or a count for test cases,
   * though the loop structure suggests it's just a placeholder to trigger reading 'inp').
   * It then reads a string of stone colors 'inp'. For each string, it iterates
   * through the characters, counts adjacent identical characters, and prints the result.
   * This structure allows processing multiple test cases from a single input stream.
   */
  // Declare 'cs' (unused, likely a placeholder or previous problem's test case count).
  // 'count' to store the number of adjacent identical stones.
  // 'inp' to store the string of stone colors.
  // 'ch' for the current character, 'last_ch' for the previous character.
  int cs, count = 0;
  string inp;
  char ch, last_ch;
  // Block Logic: Outer loop to process multiple test cases.
  // Invariant: The loop continues as long as input can be successfully read into 'cs'.
  while (cin >> cs) {
    count = 0; // Reset count for each new test case.
    cin >> inp; // Read the string of stone colors for the current test case.
    // Block Logic: Inner loop to iterate through the string and count adjacent identical stones.
    // Invariant: 'i' iterates from 0 to inp.size()-1.
    for (int i = 0; i < inp.size(); i++) {
      // For the first character, just store it as 'last_ch' and continue.
      if (i == 0) {
        last_ch = inp[i];
        continue;
      }
      ch = inp[i]; // Get the current character.
      // If the current character is the same as the last character, increment count.
      if (ch == last_ch) {
        count++;
      }
      last_ch = ch; // Update 'last_ch' to the current character for the next iteration.
    }
    // Output the total count of adjacent identical stones for the current test case.
    cout << count << endl;
  }
  return 0; // Indicate successful program execution.
}
