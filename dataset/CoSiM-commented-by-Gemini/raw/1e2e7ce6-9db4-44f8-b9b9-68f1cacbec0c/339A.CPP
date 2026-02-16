/**
 * @file 339A.CPP
 * @brief Solution for Codeforces problem 339A, "Helpful Maths".
 *
 * @description
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and rearranges it into non-decreasing order (e.g., "1+2+3").
 *
 * @algorithm
 * 1. Read the input sum as a string.
 * 2. Create a vector of integers to store the numbers.
 * 3. Iterate through each character of the input string. If the character is
 *    not a '+', convert it from a character to an integer and add it to the vector.
 * 4. Sort the vector of numbers in non-decreasing order.
 * 5. Build a new result string by iterating through the sorted vector, appending
 *    each number followed by a '+'.
 * 6. Print the resulting string, removing the final trailing '+' character.
 *
 * @time_complexity O(L log L), where L is the number of digits in the sum,
 *   dominated by the sorting step.
 * @space_complexity O(L) to store the numbers in a vector and the output string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  string out = "";
  vector<int> nums;

  // Block Logic: Parse the input string to extract numbers.
  for (char c : str) {
    int ascii = (int)c;
    // Check if the character is not a '+' (ASCII 43).
    if (ascii != 43) {
      // Convert character to integer (ASCII '0' is 48) and add to vector.
      nums.push_back(ascii - 48);
    }
  }
  // Sort the extracted numbers.
  sort(nums.begin(), nums.end());

  // Block Logic: Reconstruct the sorted sum as a new string.
  vector<int>::iterator it;
  for (it = nums.begin(); it != nums.end(); ++it) {
    out += to_string(*it);
    out += "+";
  }

  // Print the result, using substr to remove the trailing '+'.
  cout << out.substr(0, out.size() - 1) << endl;

  return 0;
}
