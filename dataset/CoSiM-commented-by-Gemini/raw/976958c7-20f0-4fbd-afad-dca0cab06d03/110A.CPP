/**
 * @file 110A.CPP
 * @brief Solution for a competitive programming problem about "nearly lucky numbers".
 * A number is defined as "nearly lucky" if the count of its lucky digits ('4' or '7')
 * is itself a lucky number (a positive integer composed only of digits '4' and '7').
 *
 * @algorithm Two-pass process:
 * 1. A linear scan of the input string to count the occurrences of '4' and '7'.
 * 2. A check to see if the resulting count is a "lucky number" by inspecting its digits.
 * @time_complexity O(L + log D), where L is the length of the input string and D is the count of lucky digits.
 * @space_complexity O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Checks if a given integer is a "lucky number".
 * A lucky number is a positive integer whose decimal representation contains
 * only the digits 4 and 7.
 * @param x The integer to check.
 * @return `true` if x is a lucky number, `false` otherwise.
 */
bool lucky(int x) {
  // A lucky number must be positive.
  if (x < 4)
    return false;
  else {
    /**
     * @block
     * @brief Iteratively checks each digit of the number.
     * The loop extracts the last digit using the modulo operator and checks
     * if it is either 4 or 7. If any other digit is found, the number is not
     * lucky.
     */
    while (x > 0) {
      if (x % 10 != 4 && x % 10 != 7)
        return false;
      x /= 10;
    }
    return true;
  }
}

/**
 * @brief Main logic to solve the "nearly lucky number" problem.
 * It reads a number as a string, counts its lucky digits ('4' and '7'), and
 * then determines if that count is itself a lucky number.
 */
void solve() {
  string s;
  cin >> s;
  int dem = 0; // Counter for lucky digits.
  int n = s.length();
  
  /**
   * @block
   * @brief Counts the number of lucky digits ('4' or '7') in the input string.
   */
  for (int i = 0; i < n; i++) {
    if (s[i] == '4' || s[i] == '7')
      dem++;
    // else
    // {
    //   cout<<"NO";
    //  exit(0);
    //}
  }

  /**
   * @block
   * @brief Checks if the count of lucky digits is a lucky number.
   */
  if (lucky(dem))
    cout << "YES";
  else
    cout << "NO";
}

int main() {
  solve();

  return 0;
}
