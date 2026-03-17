/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 339A "Helpful Maths".
 *
 * @details
 * This program takes a string representing a sum of single digits (1, 2, and 3),
 * for example, "3+2+1", and rearranges the numbers into non-decreasing order,
 * producing "1+2+3".
 *
 * @algorithm
 * The solution uses a clever trick based on ASCII character values:
 * 1. The entire input string, including the '+' characters, is read.
 * 2. The string is sorted lexicographically. Since the ASCII value of '+' is less
 *    than the ASCII values of '1', '2', and '3', the sorted string will have
 *    all '+' characters grouped at the beginning, followed by the digits in
 *    non-decreasing order (e.g., "3+2+1" becomes "++123").
 * 3. The program then iterates through the sorted string, skipping the initial
 *    '+' characters.
 * 4. It prints the digits, inserting a '+' separator between them to reconstruct
 *    the properly formatted sum.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the length of the input string.
 *   This is dominated by the `std::sort` operation.
 * - Space Complexity: O(N) to store the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  string str;
  // A flag to ensure '+' is not printed before the first number.
  // `ck` (check) is 0 for the first number, and 1 thereafter.
  int ck = 0; 
  cin >> str;

  // Sort the entire string. "3+2+1" becomes "++123".
  sort(str.begin(), str.end());

  /**
   * @brief Reconstructs the sorted sum from the sorted string.
   */
  for (int i = 0; i < str.size(); i++) {
    // Ignore the '+' characters which are now at the beginning of the string.
    if (str[i] == '+')
      continue;
    
    // If this is not the first number being printed, print a '+' separator.
    if (ck != 0)
      cout << "+";
    
    // Print the digit.
    cout << str[i];

    // Set the flag to indicate that the first number has been printed.
    ck = 1;
  }
  cout << endl;
  return 0;
}
