/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (Problem 339A).
 * @details This program reorders a string representing a sum of digits (1, 2, 3)
 * into non-decreasing order (e.g., "3+2+1" becomes "1+2+3").
 *
 * @algorithm
 * This solution uses a counting sort algorithm combined with a careful printing strategy.
 * 1. Read the input string.
 * 2. Create a frequency array `n` of size 3 to count the occurrences of '1', '2', and '3'.
 * 3. Iterate through the input string, stepping by 2 to process only the digits,
 *    and populate the frequency array. `n[0]` will hold the count of '1's, etc.
 * 4. A two-part printing process begins to handle the '+' separators correctly.
 * 5. **Part 1**: A loop finds the very first digit to be printed (i.e., the smallest
 *    digit that appeared at least once). It prints this digit and then breaks. This
 *    ensures no leading '+'.
 * 6. **Part 2**: The loop continues from where it left off. For all remaining digits
 *    to be printed, it prints a '+' before the digit itself.
 *
 * @time_complexity O(L), where L is the length of the string, due to linear passes.
 * @space_complexity O(1), as the frequency array is of constant size.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  // n[0] for '1's, n[1] for '2's, n[2] for '3's
  int n[3] = {0}, i = 0;
  cin >> s;

  /**
   * @block_logic Count the occurrences of each digit.
   */
  while (i < s.length()) {
    n[s[i] - '1']++;
    i += 2;
  }

  /**
   * @block_logic Print the first number without a leading '+'.
   * The loop finds the first non-zero count, prints the corresponding
   * digit, decrements the count, and then breaks.
   */
  for (i = 0; i < 3; i++) {
    if (n[i]--) {
      cout << i + 1;
      break;
    }
  }

  /**
   * @block_logic Print the rest of the numbers, each with a leading '+'.
   * The loop continues from where the previous one broke.
   */
  for (; i < 3; i++) {
    while (n[i]--)
      cout << "+" << i + 1;
  }
  return 0;
}