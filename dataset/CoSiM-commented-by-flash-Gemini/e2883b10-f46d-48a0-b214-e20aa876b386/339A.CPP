/**
 * @file 339A.CPP
 * @brief This program solves the "Helpful Maths" problem. It takes an input string
 * representing a sum of '1', '2', and '3' digits separated by '+' signs (e.g., "3+2+1")
 * and outputs them in non-decreasing order (e.g., "1+2+3").
 * Algorithm: Extracts digits, stores them in a vector, sorts the vector, and then
 * reconstructs the string with '+' separators.
 * Time Complexity: O(L + D log D) where L is the length of the input string and D is the number of digits.
 * The dominant factor is usually L for extraction and D log D for sorting.
 * Space Complexity: O(D) for storing the digits in a vector.
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for std::cin, std::cout.
#include <string>    // Required for std::string.
#include <vector>    // Required for std::vector.

// Using declarations to avoid repetitive std:: prefix.
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

int main() {
  string line; // Stores the input sum string.
  string output = ""; // Stores the sorted output string.
  // Reads the input sum string (e.g., "1+3+2").
  cin >> line;
  vector<int> numbers; // A vector to store the numerical values (ASCII) of the digits.
  /**
   * Block Logic: Iterates through the input string `line` to extract only the digits.
   * Characters that are not '+' are assumed to be digits ('1', '2', or '3')
   * and their ASCII values are pushed into the `numbers` vector.
   * Invariant: `numbers` vector contains the ASCII values of all digits from the input string.
   */
  for (auto chr : line) {
    if (chr != '+') {
      numbers.push_back(chr); // Store ASCII value of the digit.
    }
  }

  // Sorts the `numbers` vector in non-decreasing order. Since digits are stored as ASCII values,
  // '1' < '2' < '3', so this effectively sorts them numerically.
  std::sort(numbers.begin(), numbers.end());

  /**
   * Block Logic: Reconstructs the output string from the sorted digits.
   * It iterates through the sorted `numbers` vector, appending each digit to `output`.
   * A '+' sign is appended after each digit, except for the last one, to maintain the sum format.
   * Invariant: `output` is the sum string with digits sorted in non-decreasing order.
   */
  for (decltype(numbers.size()) i = 0; i != numbers.size(); ++i) {
    output += numbers[i]; // Append the sorted digit (ASCII value).
    if (i != numbers.size() - 1) { // If it's not the last digit.
      output += '+'; // Append a '+' separator.
    }
  }

  // Prints the final sorted sum string to standard output.
  cout << output;
  // Standard program exit.
  return 0;
}
