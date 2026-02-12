/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem.
 * @details The program takes a string representing a sum of single-digit numbers (e.g., "3+1+2")
 * and rearranges it into non-decreasing order (e.g., "1+2+3").
 *
 * @algorithm
 * 1. Read the input string which represents the sum.
 * 2. Extract the digit characters from the string, ignoring the '+' signs.
 * 3. Store the digit characters in a vector.
 * 4. Sort the vector of characters. Since ASCII values of '1' through '9' are ordered, this correctly sorts the numbers.
 * 5. Print the sorted digits, separated by '+'.
 *
 * @complexity
 * Time complexity: O(L log L), where L is the number of digits in the sum. This is dominated by the sorting step.
 * Space complexity: O(L) to store the digit characters in a vector.
 */
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  string s;
  int i;
  // This loop pattern handles multiple lines, though the problem likely involves a single line.
  while (getline(cin, s)) {
    vector<char> c;
    /**
     * @brief Extracts digits from the input string.
     * @invariant Iterates through the string with a step of 2 to skip the '+' operators.
     */
    for (i = 0; i < s.length(); i += 2) {
      c.push_back(s[i]);
    }
    
    // Sort the vector of digit characters.
    sort(c.begin(), c.end());
    
    /**
     * @brief Prints the sorted sum.
     * @invariant Prints each digit followed by a '+' except for the last one.
     */
    for (i = 0; i < c.size() - 1; i++) {
      cout << c[i] << "+";
    }
    // Print the last digit followed by a newline.
    cout << c[i] << endl;
  }
}