/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The problem requires reordering a given mathematical sum of single
 * digits (1, 2, or 3) into non-decreasing order. For example, "3+2+1" should
 * become "1+2+3".
 *
 * @algorithm
 * This solution leverages the fixed format of the input string "d+d+d...".
 * 1. Read the entire sum into a string.
 * 2. The digits are always at even-numbered indices (0, 2, 4, ...), and the
 *    '+' signs are at odd-numbered indices.
 * 3. Iterate through the string with a step of 2 to extract only the digit
 *    characters and store them in a vector.
 * 4. Sort the vector of digit characters. Since ASCII values of '1', '2', '3'
 *    are in increasing order, a standard sort is sufficient.
 * 5. Reconstruct the output string by printing the sorted digits, separated by
 *    '+' signs.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the string, dominated
 * by the sorting operation.
 * Space Complexity: O(L) to store the string and the vector of digits.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <vector>

using namespace std;

int main() {
  int i;
  string input;
  
  // This while loop will only run once due to the `return 0` inside.
  while (getline(cin, input)) {
    vector<char> num; // Vector to store the digit characters.
    
    /**
     * @block
     * @brief Extract digits by jumping over the '+' signs.
     * The loop increments by 2, picking characters at indices 0, 2, 4, etc.
     */
    for (i = 0; i < input.length(); i = i + 2) {
      num.push_back(input[i]);
    }
    
    // Sort the collected digit characters.
    sort(num.begin(), num.begin() + num.size());
    
    /**
     * @block
     * @brief Reconstruct the sorted sum string.
     */
    for (i = 0; i < num.size() - 1; i++) {
      cout << num[i] << "+";
    }
    // Print the last digit without a trailing '+'.
    cout << num[i] << endl;
    
    // Exits after the first line of input is processed.
    return 0;
  }
  return 0;
}
