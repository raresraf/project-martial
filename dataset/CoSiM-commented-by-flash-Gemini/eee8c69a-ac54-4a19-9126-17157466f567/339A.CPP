/**
 * @file 339A.CPP
 * @brief This program parses a string representing a sum of single-digit numbers
 *        (e.g., "1+2+3"). It extracts these numbers, sorts them in non-decreasing
 *        order, and then prints them back as a sum with '+' signs. This version
 *        utilizes `stringstream` for robust parsing of the input string.
 * Algorithm: String parsing using stringstream, Sorting.
 * Time Complexity: O(N log N) where N is the number of summands. Parsing is O(L)
 *                  (length of string), and sorting is O(N log N).
 * Space Complexity: O(N) for storing the extracted summands in a vector.
 */
/*
 * helpfulmaths.cpp
 * http://codeforces.com/problemset/problem/339/A
 *  Created on: Oct 25, 2017
 *      Author: athena
 * 31693931 	GNU C++14 	Accepted 	30 ms 	1876 KB
 * 2017-10-24 23:56:22 	2017-10-24 23:56:22
 */

#include <algorithm>
#include <iostream>
#include <sstream> // Required for stringstream operations.
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Parses an input string containing numbers separated by '+' into a vector of integers.
 * @param str The input string (e.g., "1+2+3").
 * @return A vector of integers representing the summands.
 * Functional Utility: Converts a string representation of a sum into its constituent integer parts.
 * Pre-condition: The input string `str` consists of single-digit numbers separated by '+'.
 */
vector<int> parseString(string str) {
  stringstream ss(str); // Creates a stringstream from the input string for parsing.
  int integer;          // Variable to store each parsed integer.
  char plus;            // Variable to consume the '+' character.
  vector<int> temp;     // Vector to store the extracted integers.

  /**
   * Block Logic: Reads integers and '+' characters from the stringstream until the end of the stream.
   * Invariant: In each iteration, an integer and a '+' (if present) are extracted, and the integer is added to 'temp'.
   */
  while (!ss.eof()) {
    ss >> integer >> plus; // Extracts an integer followed by a '+' character.
    temp.push_back(integer); // Adds the extracted integer to the vector.
  }

  return temp; // Returns the vector of summands.
}

int main() {
  string s; // Input string containing the sum.
  // Reads the input sum string.
  cin >> s;

  // Calls parseString to convert the input string into a vector of integers.
  vector<int> summands = parseString(s);

  // Sorts the vector of summands in non-decreasing order.
  sort(summands.begin(), summands.end());

  int i; // Loop counter.

  /**
   * Block Logic: Prints the sorted summands, separated by '+' signs.
   * Invariant: All but the last summand are followed by a '+'.
   *            The loop ensures correct formatting for the sum.
   */
  for (i = 0; i < summands.size() - 1; i++) { // Iterates up to the second-to-last element.
    cout << summands[i] << "+"; // Prints the summand followed by '+'.
  }
  // Prints the last summand without a trailing '+', followed by a newline.
  cout << summands[i] << endl;

  return 0;
}
