/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *        This program takes a mathematical sum consisting of numbers 1, 2, and 3
 *        separated by '+' signs, sorts the numbers in non-decreasing order,
 *        and then prints the sorted sum.
 *
 * Algorithm: Counting Sort Variation
 *   1. **Parsing (parse function):** Iterates through the input string, ignoring '+'
 *      signs, and counts the occurrences of each digit ('1', '2', '3').
 *   2. **Printing (printOutput function):** Reconstructs the sum string. It first
 *      appends all '1's, then all '2's, and finally all '3's, inserting '+' signs
 *      appropriately between numbers and between different groups of numbers
 *      (e.g., between the last '1' and the first '2' if both exist).
 *
 * Time Complexity: O(L), where L is the length of the input string `s`.
 *                  The `parse` function iterates through `s` once.
 *                  The `printOutput` function iterates a total of L characters (for string construction).
 * Space Complexity: O(L) for the input string `s` and the temporary strings used in `printOutput`.
 *                   O(1) for the `counts` array.
 */

#include <iostream>
#include <string> // Required for std::string

/**
 * @brief Parses the input sum string and counts occurrences of digits '1', '2', and '3'.
 *
 * This function iterates through the given string `s`. For every character that is
 * a digit '1', '2', or '3', it increments the corresponding counter in the `counts` array.
 * Characters that are '+' signs are ignored.
 *
 * @param s The input string representing the sum (e.g., "3+2+1").
 * @param counts An integer array of size 3, where counts[0] stores '1's,
 *               counts[1] stores '2's, and counts[2] stores '3's.
 */
void parse(std::string &s, int counts[]) {
  /**
   * Block Logic: Iterates through each character of the input string.
   * Invariant: `counts` array accurately reflects the counts of '1', '2', '3'
   *            encountered up to the current character.
   */
  for (int i = 0; i < s.size(); ++i) {
    // Conditional Logic: Increments the counter based on the digit encountered.
    if (s[i] == '1')
      ++counts[0];
    else if (s[i] == '2')
      ++counts[1];
    else if (s[i] == '3')
      ++counts[2];
  }
}

/**
 * @brief Prints the sorted sum string based on the counts of '1', '2', and '3'.
 *
 * This function reconstructs the sum string by first appending all '1's,
 * then all '2's, and then all '3's, ensuring they are separated by '+' signs
 * as per the problem's requirements.
 *
 * @param counts An integer array of size 3, containing the counts of '1's, '2's, and '3's.
 */
void printOutput(int counts[]) {
  // Temporary strings to build the parts of the sorted sum.
  std::string one = "", two = "", three = "";

  /**
   * Block Logic: Constructs the string segment for '1's.
   * Invariant: `one` contains '1's separated by '+' signs.
   */
  for (int i = 0; i < counts[0]; ++i) {
    one += '1';
    // Conditional Logic: Appends '+' unless it's the last '1'.
    if (i != counts[0] - 1)
      one += '+';
  }
  /**
   * Block Logic: Constructs the string segment for '2's.
   * Invariant: `two` contains '2's separated by '+' signs.
   */
  for (int i = 0; i < counts[1]; ++i) {
    two += '2';
    // Conditional Logic: Appends '+' unless it's the last '2'.
    if (i != counts[1] - 1)
      two += '+';
  }
  /**
   * Block Logic: Constructs the string segment for '3's.
   * Invariant: `three` contains '3's separated by '+' signs.
   */
  for (int i = 0; i < counts[2]; ++i) {
    three += '3';
    // Conditional Logic: Appends '+' unless it's the last '3'.
    if (i != counts[2] - 1)
      three += '+';
  }
  
  // Variables to hold the necessary '+' separators between groups.
  std::string c1 = "", c2 = "";
  // Conditional Logic: Determines if a '+' is needed between '1's and '2's/3's.
  if ((one != "") && ((two != "") || (three != "")))
    c1 = "+";
  // Conditional Logic: Determines if a '+' is needed between '2's and '3's.
  if ((two != "") && (three != ""))
    c2 = "+";
  
  // Prints the final sorted sum by concatenating the parts and separators.
  std::cout << one << c1 << two << c2 << three << '\n';
}

int main() {
  /**
   * @brief Main function that orchestrates the Helpful Maths problem solution.
   *
   * Reads the input sum string, parses it to count digits, and then prints
   * the digits in sorted order separated by plus signs.
   *
   * @return int Returns 0 upon successful execution.
   */
  std::string s;
  std::cin >> s; // Reads the input sum string.
  int counts[3] = {0}; // Array to store counts of '1', '2', '3'. Initialized to zeros.
  parse(s, counts);    // Calls the parse function to populate the counts.
  printOutput(counts); // Calls the printOutput function to display the sorted sum.
  return 0;
}
