/**
 * @file 339A.CPP
 * @brief This program parses a string representing a sum of '1', '2', and '3's,
 * sorts the numbers, and then prints the sum in non-decreasing order.
 * For example, "3+1+2" becomes "1+2+3".
 * Algorithm: String parsing, integer extraction, sorting, and formatted output.
 * Time Complexity: O(L log N) where L is the length of the input string and N is the count of numbers,
 *                   dominated by the sorting of numbers.
 * Space Complexity: O(N) for storing the extracted numbers in a vector.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries like iostream, string, vector, algorithm.
using namespace std;

/**
 * @brief Main function to sort a "sum" string.
 * It reads a string containing numbers '1', '2', '3' separated by '+',
 * extracts these numbers, sorts them in ascending order, and then
 * prints the sorted sum back to standard output.
 *
 * @param argc The number of command-line arguments (unused).
 * @param argv An array of command-line argument strings (unused).
 * @return 0 on successful execution.
 */
int main(int argc, char const *argv[]) {
  // @var s: String to store the input sum (e.g., "1+2+1+3").
  string s;
  // Read the input sum string from standard input.
  cin >> s;
  // @var ans: String to build the sorted sum (unused in current implementation, direct output is used).
  string ans;
  // @var arr: Vector to store the extracted numbers (1, 2, or 3).
  vector<int> arr;

  // Handle the special case where the input string has only one character (a single number).
  if (s.length() == 1) {
    cout << s; // Print the single character directly as there are no '+' signs.
  } else {
    /**
     * @brief Loop through the input string to parse numbers and skip '+' signs.
     * Invariant: `arr` contains all numbers extracted from the string `s` up to the current index `i`,
     *            excluding the '+' characters.
     */
    for (int i = 0; i < s.length(); i++) {
      // If the current character is '+', skip it and continue to the next character.
      if (s[i] == '+')
        continue;
      // If the character is '1', add the integer 1 to the vector.
      if (s[i] == '1') {
        arr.push_back(1);
      }
      // If the character is '2', add the integer 2 to the vector.
      else if (s[i] == '2') {
        arr.push_back(2);
      }
      // If the character is '3', add the integer 3 to the vector.
      else {
        arr.push_back(3);
      }
    }
    // Sort the vector of extracted numbers in non-decreasing order.
    sort(arr.begin(), arr.end());
    // @var ele: Iterator for traversing the sorted vector `arr`.
    vector<int>::iterator ele;

    /**
     * @brief Loop through the sorted numbers and print them with '+' signs in between.
     * Invariant: Numbers up to the current iterator position `ele` have been printed,
     *            separated by '+' where appropriate.
     */
    for (ele = arr.begin(); ele != arr.end(); ele++) {
      cout << *ele; // Print the current number.
      // If it's not the last element, print a '+' separator.
      if (ele != arr.end() - 1)
        cout << "+";
    }
  }

  return 0;
}
