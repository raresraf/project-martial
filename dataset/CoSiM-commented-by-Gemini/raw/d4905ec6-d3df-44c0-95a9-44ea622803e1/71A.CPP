/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 * @details This program reads an integer 'n' followed by 'n' strings. For each string, if its length is greater than 10,
 * it is abbreviated by printing its first character, the number of characters between the first and last characters,
 * and its last character. Otherwise, the string is printed as is. This is a common task in competitive programming
 * designed to handle long string inputs efficiently.
 *
 * @algorithm String abbreviation
 * @complexity
 * Time: O(N * L), where N is the number of strings and L is the maximum length of a string.
 * Since L is small and constant for practical purposes (usually strings are not excessively long in such problems),
 * it can be considered O(N).
 * Space: O(L) for storing the longest string.
 */
#include <iostream>
#include <string> // It's good practice to include <string> for std::string

using namespace std;

string a;
int n;

int main() {
  // Fast I/O is not strictly necessary for this problem due to small constraints, but it's a good habit in competitive programming.
  // std::ios_base::sync_with_stdio(false);
  // std::cin.tie(NULL);

  // Read the number of test cases.
  cin >> n;

  /**
   * @brief Main loop to process each test case.
   * This loop iterates 'n' times, processing one string in each iteration.
   */
  for (int i = 0; i < n; i++) {
    // Read the input string.
    cin >> a;

    /**
     * @brief Conditional logic for string abbreviation.
     * Pre-condition: 'a' contains the input string for the current test case.
     * Invariant: If the string length is greater than 10, it will be abbreviated.
     * Otherwise, it will be printed in its original form.
     */
    if (a.length() > 10) {
      // Abbreviate the string.
      // The format is: [first character][length-2][last character].
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    } else {
      // Print the string as is if it's not "too long".
      cout << a << endl;
    }
  }

  return 0; // Indicate successful execution.
}