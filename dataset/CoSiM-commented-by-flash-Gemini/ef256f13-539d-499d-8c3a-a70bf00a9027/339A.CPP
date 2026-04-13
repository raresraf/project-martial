/**
 * @file 339A.CPP
 * @brief This program parses a string representing a sum of single-digit numbers
 *        (e.g., "1+2+3"). It extracts the digits, sorts them in non-decreasing
 *        order, and then prints them back as a sum, separated by '+' signs.
 * Algorithm: Parsing, Sorting.
 * Time Complexity: O(N log N) due to `std::sort`, where N is the number of digits.
 *                  Parsing and printing operations are O(N).
 * Space Complexity: O(N) for storing extracted digits in a vector.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  /* Functional Utility: This commented-out block performs the same logic as the active code,
     likely representing an alternative implementation or an earlier draft.
  int i;
  string input;
  getline(cin,input);
      vector <char> numbers;
      for(i = 0; i < input.length(); i= i + 2)
      {
          numbers.push_back(input[i]);
      }
      sort(numbers.begin(),numbers.begin() + numbers.size());
      for(i = 0; i < numbers.size() - 1; i++)
      {
          cout << numbers[i] << "+";
      }
      cout << numbers[i] << endl;
*/
  string ch; // Input string, e.g., "1+2+3".
  int i, j;  // Loop counters. 'j' is unused.
  // Reads the input string.
  cin >> ch;
  vector<char> numbers; // Vector to store extracted digit characters.
  /**
   * Block Logic: Extracts digit characters from the input string.
   * Invariant: Iterates through the input string, taking every other character
   *            (skipping the '+' signs) and adding it to the 'numbers' vector.
   */
  for (i = 0; i < ch.length(); i = i + 2) {
    // Inline: Accesses the character at even indices, which are the digits.
    numbers.push_back(ch[i]);
  }
  // Sorts the extracted digit characters in non-decreasing order.
  sort(numbers.begin(), numbers.begin() + numbers.size());
  /**
   * Block Logic: Prints the sorted digit characters, separated by '+' signs.
   * Invariant: All but the last digit are followed by a '+'.
   */
  for (i = 0; i < numbers.size() - 1; i++) {
    cout << numbers[i] << "+";
  }
  // Prints the last digit without a trailing '+'.
  cout << numbers[i] << endl;
  return 0;
}
