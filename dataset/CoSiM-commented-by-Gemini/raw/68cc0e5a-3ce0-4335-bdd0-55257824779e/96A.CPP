#include <bits/stdc++.h>

using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem from Codeforces (Problem 96A), using C-style string functions.
 * @details The program determines if a football situation is "dangerous," which is defined as
 * having at least seven consecutive players from the same team. This solution uses the
 * C-style function `strstr` for the search.
 *
 * Algorithm:
 * The program reads the player configuration into a C++ `std::string`. It then checks for the
 * presence of the dangerous substrings "0000000" or "1111111". To do this, it converts the
 * `std::string` objects to C-style null-terminated strings using the `.c_str()` method.
 * The C library function `strstr` is then used to find the first occurrence of the
 * substring. If `strstr` returns a non-NULL pointer, the substring was found. If either
 * dangerous pattern is found, the program prints "YES"; otherwise, it prints "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string. The `strstr` function
 * has a linear time complexity in the worst case.
 * Space Complexity: O(L) to store the input string.
 */
int main() {
  string s;

  // checking substring problem

  cin >> s;

  string s1 = "0000000";
  string s2 = "1111111";

  /**
   * @brief Search for the dangerous patterns using `strstr`.
   * `s.c_str()` is used to get a C-style string pointer from the C++ string,
   * which is required by the `strstr` function. `strstr` returns a non-NULL
   * pointer if the substring is found.
   */
  if (strstr(s.c_str(), s1.c_str()))
    cout << "YES" << endl;
  else if (strstr(s.c_str(), s2.c_str()))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}