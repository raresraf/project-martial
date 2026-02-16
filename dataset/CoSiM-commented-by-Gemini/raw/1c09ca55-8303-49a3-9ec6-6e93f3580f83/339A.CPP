/**
 * @file 339A.CPP
 * @brief Solution for Codeforces problem 339A, "Helpful Maths".
 *
 * This program rearranges a given mathematical sum of single-digit numbers
 * (1, 2, and 3) into non-decreasing order. For example, an input of "3+2+1"
 * would be output as "1+2+3".
 *
 * @algorithm
 * 1. Read the input sum as a string.
 * 2. Create an integer vector to store the numbers from the sum.
 * 3. Iterate through the input string with a step of 2 to skip the '+' signs.
 * 4. Convert the character digits to integers and store them in the vector.
 * 5. Sort the vector in non-decreasing order.
 * 6. Reconstruct the output string by iterating through the sorted vector,
 *    printing each number followed by a '+' sign, except for the last number.
 *
 * @time_complexity O(L log L), where L is the number of digits in the sum.
 *   The complexity is dominated by the sorting step.
 * @space_complexity O(L) to store the numbers in a vector.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  vector<int> v;

  // Block Logic: Parse the input string to extract the numbers.
  // The loop iterates with a step of 2 to select characters that are digits,
  // skipping the '+' operators.
  for (int i = 0; i < str.size(); i = i + 2) {
    // Convert character to integer ('0' has ASCII value 48) and add to vector.
    v.push_back((int)str[i] - 48);
  }

  // Sort the numbers in non-decreasing order.
  sort(v.begin(), v.end());

  // Block Logic: Reconstruct and print the sorted mathematical sum.
  // The loop prints each number followed by a '+' separator.
  for (int i = 0; i < v.size() - 1; i++) {
    cout << v[i] << "+";
  }
  // Print the last number without a trailing '+'.
  cout << v[v.size() - 1];
  return 0;
}
