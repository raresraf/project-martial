/**
 * @file raw/c5f4313f-bc6d-41f6-a77e-6ee68bdef25e/339A.CPP
 * @brief Rearranges a sum of single-digit numbers into non-decreasing order.
 *
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and outputs the sum with the numbers rearranged in
 * non-decreasing order (e.g., "1+2+3").
 *
 * @algorithm
 * 1. Read the input string containing the sum.
 * 2. Extract the digit characters by iterating through the string with a step of 2,
 *    skipping the '+' operators.
 * 3. Store the ASCII values of these digits in an integer array.
 * 4. Sort the integer array. Since ASCII values of digits '0' through '9' are
 *    sequential, this correctly sorts the numbers themselves.
 * 5. Reconstruct the output string by iterating through the sorted array,
 *    printing each number (converted from ASCII) and separating them with '+'.
 *
 * @time_complexity O(N log N), where N is the number of summands in the input string.
 *                  The complexity is dominated by the sorting operation.
 * @space_complexity O(N) to store the numbers extracted from the string.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  bool flag = false;
  string s;
  cin >> s;
  int n[s.length() / 2 + 1];

  /**
   * @block
   * This loop parses the input string `s` to extract the numbers.
   * It iterates with a step of 2 to skip the '+' signs.
   */
  for (int i = 0; i < s.length(); i += 2) {
    // Inline: Stores the character's ASCII value. '1' becomes 49, '2' becomes 50, etc.
    n[i / 2] = s[i];
  }

  // Sorts the array of numbers. Since they are stored as ASCII values,
  // this correctly sorts them numerically (e.g., 49 < 50, so '1' < '2').
  sort(n, n + (s.length() / 2 + 1));

  /**
   * @block
   * This loop reconstructs and prints the sorted sum.
   */
  for (int i = 0; i < s.length() / 2 + 1; i++) {
    // Prepend a '+' before every number except the first one.
    if (flag)
      cout << "+";
    flag = true;
    // Inline: Converts the ASCII value back to an integer by subtracting
    // the ASCII value of '0' (48) before printing.
    cout << n[i] - 48;
  }
}
