/**
 * @file 339A.CPP
 * @brief This program solves the "Xenia and Bushes" problem (Codeforces 339A).
 *   It rearranges a given sum of numbers (1, 2, or 3) into non-decreasing order.
 *   The input is a string representing the sum (e.g., "1+3+2+1"), and the output should be the rearranged sum (e.g., "1+1+2+3").
 *   Problem description: https://codeforces.com/problemset/problem/339/A
 * * Algorithm: Frequency counting and string building.
 *   1. It counts the occurrences of '1', '2', and '3' in the input sum string.
 *   2. It then constructs the output string by appending all '1's, then all '2's, then all '3's,
 *      interspersed with '+' signs.
 *   3. Finally, it removes the trailing '+' sign.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass for
 *   frequency counting and then passes proportional to the number of digits for string building.
 * Space Complexity: O(L) to store the input and output strings.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
#include <iostream>  // Required for std::cin, std::cout.
#include <string>    // Required for std::string.
using namespace std;

/**
 * @brief Main function to rearrange and print the sum in non-decreasing order.
 * Functional Utility: Reads the input sum string, counts the frequencies of digits '1', '2', '3',
 *                     then constructs and prints the sorted sum string with '+' separators.
 */
int main() {
  string input, output; // `input` stores the original string, `output` builds the sorted sum.
  // `count1`, `count2`, `count3`: Frequency counters for digits '1', '2', and '3' respectively.
  int count1 = 0, count2 = 0, count3 = 0;
  // Block Logic: Reads the input sum string.
  cin >> input;
  /**
   * Block Logic: Iterates through the input string to count the occurrences of each digit ('1', '2', '3').
   * Pre-condition: `input` contains digits '1', '2', '3' and '+' signs.
   * Invariant: `count1`, `count2`, `count3` accurately reflect the frequencies of their respective digits.
   */
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == '1')
      count1++;
    else if (input[i] == '2')
      count2++;
    else if (input[i] == '3')
      count3++;
  }
  int i; // Loop counter, re-used.
  /**
   * Block Logic: Appends '1's to the `output` string based on `count1`.
   * Invariant: Each '1' is followed by a '+', except for the last '1' that might be added.
   * Inline: `count1 > 0` check ensures that the loop only runs if there are '1's to append.
   */
  for (i = 0; i < count1 && count1 > 0; i++)
    output += "1+";
  /**
   * Block Logic: Appends '2's to the `output` string based on `count2`.
   * Invariant: Each '2' is followed by a '+'.
   */
  for (i = 0; i < count2 && count2 > 0; i++)
    output += "2+";
  /**
   * Block Logic: Appends '3's to the `output` string based on `count3`.
   * Invariant: Each '3' is followed by a '+'.
   */
  for (i = 0; i < count3 && count3 > 0; i++)
    output += "3+";
  // Block Logic: Prints the final `output` string, removing the last character (which is a superfluous '+').
  // Inline: `output.substr(0, output.length() - 1)` extracts a substring from the beginning
  //         up to, but not including, the last character.
  cout << output.substr(0, output.length() - 1);
  return 0;
}
