/**
 * @file 339A.CPP
 * @brief This program processes a string representing a sum of single-digit numbers
 *        (specifically '1', '2', and '3'), counts the occurrences of each digit,
 *        and then prints them in non-decreasing order, separated by '+' signs.
 * Algorithm: Counting Sort (implicit). Digits are counted, then printed in sorted order.
 * Time Complexity: O(L) where L is the length of the input string, due to single passes
 *                  for counting and printing.
 * Space Complexity: O(1) for storing fixed counters for each digit.
 */
#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin); // Commented out: redirects standard input from "input.txt"
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios_base::sync_with_stdio(false);
  string s; // Input string containing the sum.
  // Reads the input sum string.
  cin >> s;
  // count: Tracks the number of digits remaining to be printed, used for correct '+' placement.
  // one, two, three: Counters for digits '1', '2', and '3' respectively.
  int count = 0;
  int one = 0, two = 0, three = 0;
  /**
   * Block Logic: Iterates through the input string to count occurrences of digits '1', '2', and '3'.
   * Invariant: For each digit encountered, its respective counter and the total 'count' are incremented.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '1') {
      ++count; // Increments total digit count.
      ++one;   // Increments count for digit '1'.
    }
    if (s[i] == '2') {
      ++count; // Increments total digit count.
      ++two;   // Increments count for digit '2'.
    }
    if (s[i] == '3') {
      ++count; // Increments total digit count.
      ++three; // Increments count for digit '3'.
    }
  }
  /**
   * Block Logic: Prints all '1's.
   * Invariant: Prints '1' followed by '+' if there are more digits to print, otherwise prints '1' followed by newline.
   */
  for (int i = 0; i < one; i++) {
    count -= 1; // Decrements the remaining digit count.
    if (count == 0)
      cout << "1" << endl; // Last digit, print newline.
    else
      cout << "1+"; // Not the last digit, print '+'.
  }
  /**
   * Block Logic: Prints all '2's.
   * Invariant: Prints '2' followed by '+' if there are more digits to print, otherwise prints '2' followed by newline.
   */
  for (int i = 0; i < two; i++) {
    count -= 1; // Decrements the remaining digit count.
    if (count == 0)
      cout << "2" << endl; // Last digit, print newline.
    else
      cout << "2+"; // Not the last digit, print '+'.
  }
  /**
   * Block Logic: Prints all '3's.
   * Invariant: Prints '3' followed by '+' if there are more digits to print, otherwise prints '3' followed by newline.
   */
  for (int i = 0; i < three; i++) {
    count -= 1; // Decrements the remaining digit count.
    if (count == 0)
      cout << "3" << endl; // Last digit, print newline.
    else
      cout << "3+"; // Not the last digit, print '+'.
  }
}