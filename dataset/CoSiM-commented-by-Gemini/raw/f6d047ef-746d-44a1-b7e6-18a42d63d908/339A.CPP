/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (339A).
 * @details The program parses a string representing a sum (e.g., "1+3+2"),
 * sorts the numbers, and prints the sorted sum (e.g., "1+2+3").
 *
 * Algorithm: Stringstream Parsing and Sorting.
 * 1. Read the entire input line as a string.
 * 2. Initialize a `stringstream` with the input string.
 * 3. Use `getline` with the '+' delimiter to tokenize the string, extracting each number.
 * 4. Convert each number token from string to integer using `stoi` and store it in a vector.
 * 5. Sort the vector of integers in non-decreasing order.
 * 6. Print the elements of the sorted vector, separated by '+'.
 *
 * Time Complexity: O(L + N log N), where L is the length of the string and N is the
 * number of digits. L for parsing and N log N for sorting.
 * Space Complexity: O(N) to store the vector of integers.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int cnt = 0;
  vector<int> vec;
  // Read the entire line containing the sum.
  getline(cin, s);
  // Use a stringstream for easy parsing of the delimited string.
  stringstream ss(s);
  string ind;
  /**
   * Block Logic: Tokenize the input string by the '+' delimiter.
   * `getline` extracts substrings between the '+' characters.
   */
  while (getline(ss, ind, '+')) {
    // Convert the numeric string token to an integer.
    int a = stoi(ind);
    vec.push_back(a);
  }
  // Sort the vector of numbers in ascending order.
  sort(vec.begin(), vec.end());

  // Print the first element separately to handle the '+' logic easily.
  cout << vec[0];
  /**
   * Block Logic: Print the rest of the sorted numbers, each prefixed with a '+'.
   */
  for (int i = 1; i < vec.size(); i++) {
    cout << "+" << vec[i];
  }
  cout << endl;

  return 0;
}
