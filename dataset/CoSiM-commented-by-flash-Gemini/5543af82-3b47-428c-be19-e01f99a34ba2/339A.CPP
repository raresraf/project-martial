/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A - "Helpful Maths".
 * This program sorts the terms of a sum given as a string. The sum consists
 * of numbers 1, 2, or 3, separated by '+' signs. The output should be
 * the same sum with its terms sorted in non-decreasing order.
 * Algorithm: Parse the string to extract numbers, sort the numbers, then print them.
 * Time Complexity: O(L log L), where L is the length of the input string.
 *                  This is dominated by the sorting step.
 * Space Complexity: O(L) - to store the extracted numbers in a `std::vector<char>`.
 */
#include <algorithm> // Required for std::sort.
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, string, vector).
#include <vector>    // Required for std::vector.

using namespace std; // Uses the standard namespace.

int main() {
  int i;         // Loop counter variable.
  string input;  // Declares a string to store the input sum.
  
  /**
   * Block Logic: Reads the input sum string line by line and processes it.
   * The `while (getline(cin, input))` loop is used to read an entire line,
   * but the `return 0;` inside makes it process only the first line of input.
   */
  while (getline(cin, input)) { // Reads a line from standard input into 'input' string.
    vector<char> num; // Declares a vector of characters to store the extracted digits.
    
    /**
     * Block Logic: Extracts digit characters from the input string.
     * It iterates through the string, taking every second character to skip '+' signs.
     */
    for (i = 0; i < input.length(); i = i + 2) { // Loop, incrementing by 2 to skip '+' characters.
      num.push_back(input[i]); // Adds the digit character to the vector.
    }
    
    // Sorts the vector of digit characters in non-decreasing order.
    sort(num.begin(), num.begin() + num.size());
    
    /**
     * Block Logic: Prints the sorted digit characters, separated by '+', to form the sorted sum.
     * Ensures that no '+' is printed after the last digit.
     */
    for (i = 0; i < num.size() - 1; i++) { // Loop up to the second-to-last digit.
      cout << num[i] << "+"; // Prints the digit and a '+' separator.
    }
    cout << num[i] << endl; // Prints the last digit, followed by a newline.
    
    return 0; // Exits the program after processing the first line of input.
  }
  return 0; // Should not be reached if input is valid, acts as a fallback.
}
