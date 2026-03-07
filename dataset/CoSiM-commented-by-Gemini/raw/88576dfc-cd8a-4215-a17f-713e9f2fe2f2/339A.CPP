/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem using a vector and std::sort.
 *
 * @details
 * The program reorders a string representing a sum (e.g., "3+2+1") so that the
 * terms are in non-decreasing order (e.g., "1+2+3"). This is a straightforward
 * implementation that extracts the numbers, sorts them, and then reconstructs
 * the string.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a `std::vector` of characters to hold the digits.
 * 3. Iterate through the input string and push all characters that are not '+'
 *    into the vector.
 * 4. Sort the vector of characters. Since the ASCII values of '1', '2', '3' are
 *    in increasing order, this correctly sorts the numbers.
 * 5. Iterate through the sorted vector and print each character, separated by '+'.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum,
 *                  dominated by the `std::sort` operation.
 * Space Complexity: O(L) to store the digits in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

// A standard comparison function, though not strictly necessary as std::sort on
// chars would produce the same result by default.
bool compare(int i, int j) { return i < j; }
int main() {
  string s;

  cin >> s;

  vector<char> inputs;

  /**
   * @brief Extracts the digit characters from the input string.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      inputs.push_back(s[i]);
  }

  // Sort the vector of digit characters.
  sort(inputs.begin(), inputs.end(), compare);

  /**
   * @brief Prints the sorted digits, separated by '+'.
   */
  for (int i = 0; i < inputs.size(); i++) {
    // Special handling for the last element to avoid a trailing '+'.
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
