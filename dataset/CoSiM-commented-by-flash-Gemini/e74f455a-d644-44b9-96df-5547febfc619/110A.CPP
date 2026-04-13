/**
 * @file 110A.CPP
 * @brief This program determines if a given number (provided as a string)
 *        is a "nearly lucky" number. A number is considered nearly lucky
 *        if the count of its 'lucky' digits (4s and 7s) is itself composed
 *        entirely of lucky digits.
 * Algorithm: Digit counting, String conversion of count, Digit-by-digit validation.
 * Time Complexity: O(L + log_10(C)) where L is the length of the input string
 *                  and C is the count of lucky digits. The string traversal is O(L),
 *                  and converting and checking the count is proportional to the number
 *                  of digits in the count.
 * Space Complexity: O(L + log_10(C)) for storing the input string and the string
 *                   representation of the lucky digit count.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> ordered_set;

int a[105], b[105]; // Unused arrays, possibly remnants from a template.
int main() {
  string s; // Input number as a string.
  int i, cp = 0; // i: loop counter. cp: count of lucky digits ('4' or '7').
  // Reads the input number as a string.
  cin >> s;
  int sz = s.length(); // Stores the length of the input string.
  /**
   * Block Logic: Iterates through each character (digit) of the input string
   *              and increments `cp` if the digit is '4' or '7'.
   * Invariant: `cp` stores the total occurrences of lucky digits in the number.
   */
  for (i = 0; i < sz; ++i) {
    if (s[i] == '4' || s[i] == '7') { // Checks if the current digit is '4' or '7'.
      cp++;                         // Increments the lucky digit counter.
    }
  }
  // cout << cp << endl; // Commented out debug print of the lucky digit count.

  // Converts the count of lucky digits (`cp`) to its string representation.
  string str = to_string(cp);
  /**
   * Block Logic: Iterates through each character (digit) of the string representation
   *              of the lucky digit count (`str`) to validate if all its digits are '4' or '7'.
   * Invariant: The loop breaks at the first non-lucky digit or completes if all digits are lucky.
   */
  for (i = 0; i < str.length(); ++i) {
    if (str[i] == '4' || str[i] == '7') { // Checks if the current digit of the count is '4' or '7'.
      continue;                            // Continues to the next digit if lucky.
    } else
      break; // Breaks if a non-lucky digit is found.
  }
  /**
   * Block Logic: Determines if the number is "nearly lucky" based on the validation loop.
   * Invariant: Prints "YES" if all digits of `cp` were lucky (loop completed), else "NO".
   */
  if (i == str.length()) // If the loop completed, meaning all digits of 'cp' were lucky.
    cout << "YES" << endl; // The number is nearly lucky.
  else
    cout << "NO" << endl; // The number is not nearly lucky.
}