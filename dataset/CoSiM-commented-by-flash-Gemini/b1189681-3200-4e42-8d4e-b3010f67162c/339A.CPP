/**
 * @file 339A.CPP
 * @brief This program solves the "Xenia and Bushes" problem (Codeforces 339A).
 *   It rearranges a given sum of numbers (1, 2, or 3) into non-decreasing order.
 *   The input is a string representing the sum (e.g., "1+3+2+1"), and the output should be the rearranged sum (e.g., "1+1+2+3").
 *   Problem description: https://codeforces.com/problemset/problem/339/A
 * * Algorithm: String building with three accumulators and conditional output.
 *   1. It iterates through the input string, extracting digits ('1', '2', '3')
 *      and appending them with a '+' suffix to separate accumulator strings (`w1`, `w2`, `w3`).
 *   2. It then combines and prints these accumulator strings, carefully handling the removal
 *      of any trailing '+' characters, to produce the final sorted sum.
 * Time Complexity: O(L) where L is the length of the input string. The parsing loop
 *   and string concatenations/substring operations are proportional to L.
 * Space Complexity: O(L) to store the input string and the three accumulator strings (`w1`, `w2`, `w3`).
 */
#include <iostream> // Required for std::cin, std::cout, std::endl.
#include <string>   // Required for std::string and its methods.
using namespace std;

/**
 * @brief Main function to rearrange and print the sum in non-decreasing order.
 * Functional Utility: Reads the input sum string, categorizes digits into separate
 *                     strings, and then combines and prints them in sorted order.
 */
int main() {
  // `word`: Stores the original input string.
  // `w1`, `w2`, `w3`: Accumulator strings for '1's, '2's, and '3's respectively, with '+' suffixes.
  string word, w1 = "", w2 = "", w3 = "";
  // Block Logic: Reads the input sum string.
  cin >> word;
  // `size`: Calculates the number of digits in the input string (approximately half the length + 1).
  // `temp`: Index for iterating through the input `word` to extract digits.
  int size = word.size() / 2 + 1, temp = 0;

  /**
   * Block Logic: Iterates through the input `word`, extracting digits and appending them to
   *              their respective accumulator strings (`w1`, `w2`, `w3`) with a '+' suffix.
   * Pre-condition: `word` contains digits '1', '2', '3' and '+' signs.
   * Invariant: `w1`, `w2`, `w3` accumulate the digits in order of appearance, each followed by '+'.
   */
  while (size--) {
    // Conditional Logic: Classifies the digit at `word[temp]` and appends to the correct string.
    if (word[temp] == '1')
      w1 += "1+";
    else if (word[temp] == '2')
      w2 += "2+";
    else // Must be '3'.
      w3 += "3+";

    temp += 2; // Advances `temp` by 2 to skip the digit and the '+' character.
  }

  /**
   * Conditional Logic: Complex block for printing the combined and sorted sum,
   *                    handling the removal of trailing '+' characters.
   * This logic ensures '1's are printed first, then '2's, then '3's, and correctly
   * removes the final '+' from the entire output.
   */
  if (w2.size() > 0 || w3.size() > 0) { // If there are any '2's or '3's.
    cout << w1;                       // Prints all '1's (with their trailing '+').
    if (w3.size() == 0)             // If there are no '3's.
      // Prints all '2's, removing the trailing '+' (since it's the last part).
      cout << w2.substr(0, w2.size() - 1);
    else
      cout << w2; // Prints all '2's (with their trailing '+').
  } else
    // If only '1's are present, prints `w1` removing its trailing '+'.
    cout << w1.substr(0, w1.size() - 1);

  // Block Logic: Prints `w3`, ensuring the trailing '+' is removed.
  // This statement is executed regardless of previous `if-else` branches,
  // making the logic for `w1` and `w2` in the first `if-else` block potentially flawed
  // if `w3` is empty or if its trailing '+' is not intended to be removed independently.
  cout << w3.substr(0, w3.size() - 1) << endl;
  return 0;
}
