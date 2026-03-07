/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem that calculates a value based on
 * a predefined mapping of digits and the first two digits of an input string.
 * This problem likely involves a system where each digit has an associated "cost" or "segment count"
 * (e.g., for a 7-segment display), and the task is to multiply the costs of the first two digits.
 * Algorithm: Digit mapping and multiplication.
 * Time Complexity: O(1) after initial setup, as it processes only two digits.
 * Space Complexity: O(1) for storing the digit mapping.
 */
#include <algorithm> // Included but not used.
#include <climits>   // Included but not used.
#include <cmath>     // Included but not used.
#include <cstdio>    // Included but not used.
#include <cstring>   // Included but not used.
#include <iostream>  // For input/output operations.
#include <map>       // Included but not used.
#include <queue>     // Included but not used.
#include <set>       // Included but not used.
#include <sstream>   // Included but not used.
#include <vector>    // Used for `vi digit(10)`.
using namespace std;

typedef vector<int> vi;   // Shorthand for vector of integers (used for 'digit').
typedef vector<vi> vvi;   // Shorthand for vector of vectors of integers (unused).

// Global vector `digit` to store the mapped values for each digit (0-9).
// Its size is 10, initialized with default values (0).
vi digit(10);

int main() {
  // Block Logic: Initializes the `digit` vector with predefined values.
  // These values likely represent a specific property of each digit,
  // e.g., number of segments in a 7-segment display or some custom cost.
  digit[0] = 2; // Value for digit 0.
  digit[1] = 7; // Value for digit 1.
  digit[2] = 2; // Value for digit 2.
  digit[3] = 3; // Value for digit 3.
  digit[4] = 3; // Value for digit 4.
  digit[5] = 4; // Value for digit 5.
  digit[6] = 2; // Value for digit 6.
  digit[7] = 5; // Value for digit 7.
  digit[8] = 1; // Value for digit 8.
  digit[9] = 2; // Value for digit 9.

  string s; // Declares 's' to store the input string of digits.
  // Reads the input string from standard input.
  cin >> s;
  
  // Calculates the product of the mapped values of the first two digits of the input string.
  // `s[0] - '0'` converts the character digit '0'-'9' to its integer value.
  // `digit[...]` retrieves the corresponding mapped value.
  cout << digit[(s[0] - '0')] * digit[(s[1] - '0')] << endl;
  
  return 0; // Indicates successful program execution.
}