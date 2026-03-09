/**
 * @file 339A.CPP
 * @brief Solution to Codeforces Problem 339A: "Helpful Maths".
 *        This program takes a string representing a sum of numbers (e.g., "1+1+3+1")
 *        and reorders the numbers in non-decreasing order while preserving the '+' signs.
 *
 * Algorithm: Digit extraction, sorting, and formatted output. It filters out non-digit
 *            characters, collects the digits, sorts them, and then reconstructs the
 *            sum expression with sorted digits.
 * Time Complexity: O(L + D log D), where L is the length of the input string and D is
 *                  the number of digits extracted. The dominant factor is usually L
 *                  for extraction and D log D for sorting the digits.
 * Space Complexity: O(D) for storing the extracted digits in a vector.
 */

#include <algorithm> // Required for std::sort.
#include <bits/stdc++.h> // A common competitive programming header, includes many standard libraries.
using namespace std;

// This function `inc` is declared but not used in the main logic of the program.
// It seems to be an artifact or an incomplete thought for a custom comparison function.
bool inc(int i, int j) { return (i < j); }

int main() {
  /**
   * Functional Utility: Reads an arithmetic expression string, extracts its numbers,
   *                     sorts them, and prints the sorted expression.
   */
  string s; // Stores the input arithmetic expression.
  cin >> s; // Reads the expression from standard input.

  vector<char> v; // A dynamic array to store the extracted digit characters.
  int n = s.length(); // Stores the length of the input string for loop iteration.

  /**
   * Block Logic: Iterates through the input string to extract all digit characters.
   * Pre-condition: `s` contains the input string; `v` is empty.
   * Invariant: After each iteration, `v` contains all digit characters encountered
   *            in `s` up to the current index `i`.
   */
  for (int i = 0; i < n; i++) {
    if (isdigit(s[i])) { // Condition Check: Checks if the current character is a digit.
      v.push_back(s[i]); // Adds the digit character to the vector.
    }
  }

  // Algorithm Core: Sorts the extracted digit characters in non-decreasing order.
  // The default `std::sort` for characters will sort them based on their ASCII values,
  // which for digits '0' through '9' correctly corresponds to their numerical order.
  sort(v.begin(), v.end());

  int size_v = v.size(); // Stores the number of extracted digits for output loop.

  /**
   * Block Logic: Prints the sorted digit characters, separated by '+' signs.
   * Pre-condition: `v` contains sorted digit characters; `size_v` is the count of these digits.
   * Invariant: After each iteration, a digit (and potentially a '+') has been printed.
   *            The last digit is printed without a trailing '+'.
   */
  for (int i = 0; i < size_v; i++) {
    if (i == size_v - 1)
      cout << v[i]; // If it's the last digit, print without '+'.
    else
      cout << v[i] << "+"; // Otherwise, print digit followed by '+'.
  }

  return 0; // Indicates successful program termination.
}
