/**
 * @file 339A.CPP
 * @brief Reorders single-digit numbers ('1', '2', '3') in a mathematical sum string into non-decreasing order.
 *
 * Problem Name: Helpful Maths (Codeforces 339A)
 * Algorithm: Extracts numerical digits from the input string, stores them in a vector, sorts the vector,
 *            and then reconstructs the sum string with numbers in non-decreasing order.
 * Time Complexity: O(L + N log N), where L is the length of the input string and N is the count of numbers.
 *                  Extraction takes O(L), sorting takes O(N log N).
 * Space Complexity: O(N) for storing the extracted numbers in a `std::vector`.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>, <vector>, <algorithm>.
#include <iostream>      // Explicitly included for std::cin, std::cout.
#include <string.h>      // Not strictly necessary for std::string but included in original.

using namespace std;

/**
 * @brief Main function to handle input, process the string, and print the sorted sum.
 * @pre Input consists of a single string representing a sum of single-digit numbers ('1', '2', '3') separated by '+'.
 * @post Prints the sum string with numbers reordered in non-decreasing order (e.g., "1+1+2+3").
 */
int main() {
  vector<int> s; // Vector to store the extracted single-digit numbers.
  string a;      // Variable to store the input sum string.
  // Functional Utility: Reads the input sum string from standard input.
  cin >> a;

  /**
   * Block Logic: Iterates through the input string to extract numerical digits ('1', '2', '3').
   * Invariant: Only '1', '2', or '3' characters are converted to integers and added to the 's' vector.
   *            Other characters (like '+') are ignored.
   */
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == '1')
      s.push_back(1); // Adds 1 to the vector.
    else if (a[i] == '2')
      s.push_back(2); // Adds 2 to the vector.
    else if (a[i] == '3')
      s.push_back(3); // Adds 3 to the vector.
    else
      cout << ""; // Does nothing for other characters (e.g., '+'), but prints an empty string if it encounters 'else'.
                  // This cout << "" is a no-op but will be preserved due to no-code-mutation rule.
  }

  // Functional Utility: Sorts the extracted numbers in non-decreasing order.
  sort(s.begin(), s.end());

  int w = 0; // Index for printing elements of the vector.

  /**
   * Block Logic: Prints the sorted numbers, inserting '+' signs between all but the last number.
   * Invariant: Each number from the vector (except the last one) is printed followed by a '+'.
   */
  for (auto it = s.begin(); it != (s.end() - 1); ++it) { // Iterates until the second to last element.
    cout << *it << "+";                                // Prints the number and a '+'.
    w++;                                                // Increments 'w' to track the position.
  }
  // Functional Utility: Prints the last number without a trailing '+'.
  cout << s[w]; // Prints the last element using the 'w' index.
  // Indicates successful program execution.
  return 0;
}
