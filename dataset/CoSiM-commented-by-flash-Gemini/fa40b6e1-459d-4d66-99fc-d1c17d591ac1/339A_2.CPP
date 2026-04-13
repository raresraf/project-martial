/**
 * @file 339A_2.CPP
 * @brief This program solves the "Helpful Maths" problem. It takes an input string
 * representing a sum of '1', '2', and '3' digits separated by '+' signs (e.g., "3+2+1")
 * and outputs them in non-decreasing order (e.g., "1+2+3").
 * Algorithm: Parses the input string to extract individual numbers, stores them in a vector,
 * sorts the vector, and then reconstructs the string by concatenating the sorted numbers
 * with '+' signs in between.
 * Time Complexity: O(L + D log D) where L is the length of the input string and D is the number of digits.
 * Dominant factors are string parsing (L) and sorting (D log D).
 * Space Complexity: O(D) for storing the digits in a vector and intermediate strings.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.

using namespace std;

int main() {
  vector<int> vec;  // A vector to store the extracted integer values of the digits.
  string s;         // Stores the input sum string (e.g., "1+3+2").
  string tmp;       // Temporary string to build up numbers before conversion to int.
  // Reads the input sum string.
  cin >> s;
  int len = s.size(); // Stores the length of the input string.
  int left = 0;       // Index to track the current position in the input string.

  /**
   * Block Logic: Parses the input string `s` to extract numbers separated by '+'.
   * It iterates through the string character by character. If a character is not '+',
   * it's part of a number and is appended to `tmp`. If it's '+', the accumulated `tmp`
   * is converted to an integer and pushed into `vec`, and `tmp` is reset.
   */
  while (left < len) {
    if (s[left] != '+') {
      // Creates a C-style character array of size 2. `s[left]` holds the digit, `0` is the null terminator.
      char p[2] = {s[left], 0};
      string t = p; // Converts the C-style char array to a std::string.
      tmp += t;     // Appends the digit string to `tmp`.
      left++;       // Moves to the next character.
    } else {
      int tt = stoi(tmp);       // Converts the accumulated digit string `tmp` to an integer.
      vec.push_back(tt);        // Adds the integer to the vector.
      tmp = "";                 // Resets `tmp` for the next number.
      left++;                   // Moves past the '+' character.
    }
  }
  // Handles the last number in the string, which is not followed by a '+'.
  char t1[2] = {s[s.size() - 1], 0}; // Extracts the last character (digit).
  string sss = t1;                   // Converts it to a string.

  int ttt = stoi(sss);      // Converts the last digit string to an integer.
  vec.push_back(ttt);       // Adds the last integer to the vector.

  string ret = "";          // Initializes an empty string to build the sorted output.
  // Sorts the vector of extracted numbers in non-decreasing order.
  sort(vec.begin(), vec.end());
  /**
   * Block Logic: Reconstructs the sorted sum string from the sorted numbers in `vec`.
   * Each number is converted back to a string and appended to `ret`, followed by a '+'.
   */
  for (auto i : vec) {
    ret += to_string(i) + '+';
  }
  // Removes the trailing '+' character from the end of the `ret` string.
  ret = ret.substr(0, ret.size() - 1);
  // Prints the final sorted sum string to standard output.
  cout << ret;
  // Standard program exit.
  return 0;
}