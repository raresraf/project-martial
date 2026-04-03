/**
 * @file 339A.CPP
 * @brief Sorts the summands of a mathematical expression.
 *
 * This program reads a string representing a sum of single-digit numbers (e.g., "1+3+2"),
 * parses the numbers, sorts them in non-decreasing order, and prints the resulting
 * expression.
 *
 * @algorithm Sorting-based reordering.
 * @time_complexity O(L log L) where L is the length of the input string, dominated by the sort operation.
 * @space_complexity O(L) to store the digits from the input string.
 */
#include <bits/stdc++.h>
using namespace std;

// @brief A custom comparator function for sorting integers in ascending order.
// Although the vector contains chars, they are promoted to ints for comparison,
// so this correctly sorts the digits numerically.
bool compare(int i, int j) { return i < j; }
int main() {
  string s;

  // Functional Utility: Reads the entire line of input representing the sum.
  cin >> s;

  vector<char> inputs;

  /**
   * @brief Block Logic: Parses the input string to extract the summands.
   *
   * This loop iterates through the input string `s` and collects all characters
   * that are not '+' into a vector of characters. Each character represents a
   * single-digit number.
   * @invariant At the end of the loop, `inputs` contains all the digits from `s`.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      inputs.push_back(s[i]);
  }

  /**
   * @brief Block Logic: Sorts the extracted digits.
   *
   * Utilizes the std::sort algorithm to arrange the digits in non-decreasing
   * order. A custom comparison function `compare` is provided, although the
   * default `operator<` for characters would have yielded the same result for digits.
   */
  sort(inputs.begin(), inputs.end(), compare);

  /**
   * @brief Block Logic: Reconstructs and prints the sorted sum expression.
   *
   * This loop iterates through the sorted vector of digits and prints them,
   * interleaved with '+' symbols.
   * @invariant The digits are printed in sorted order. A '+' is printed after
   * each digit except for the last one.
   */
  for (int i = 0; i < inputs.size(); i++) {
    // Block Logic: Handles the last element of the vector.
    // Pre-condition: `i` is the index of the current digit.
    // Invariant: If it's the last digit, it's printed followed by a newline
    // instead of a '+', correctly terminating the expression.
    if (inputs.size() - 1 == i) {
      cout << inputs[i] << endl;
      break;
    }
    cout << inputs[i] << '+';
  }
  return 0;
}

/**
3+2+1
output:
1+2+3

1+1+3+1+3
output:
1+1+1+3+3

2
output:
2
**/