/**
 * @file 96A.CPP
 * @brief Solution for Codeforces problem 96A, "Football".
 *
 * @description
 * This program determines if a situation in a football match is "dangerous".
 * A situation is dangerous if there are at least 7 consecutive players of the
 * same team. The input is a string of '0's and '1's representing the players.
 * The program checks for the presence of "0000000" or "1111111" as a substring.
 *
 * @algorithm
 * 1. Read the input string `s`.
 * 2. Define two target strings: "1111111" and "0000000".
 * 3. Use the C-style function `strstr` to search for the occurrence of either
 *    target string within the input string.
 * 4. `strstr` requires C-style strings, so `s.c_str()` is used for the conversion.
 * 5. If `strstr` finds either substring, it returns a non-null pointer (true),
 *    indicating a dangerous situation.
 * 6. Print "YES" if dangerous, "NO" otherwise.
 *
 * @note A more idiomatic C++ approach would be to use `std::string::find`.
 *
 * @time_complexity O(L), where L is the length of the input string. `strstr`
 *   has a linear time complexity on average.
 * @space_complexity O(1), ignoring the storage for the input string.
 */
#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  cin >> s;
  // `t` and `o` represent the two "dangerous" situations.
  string t = "1111111";
  string o = "0000000";

  // This commented-out line suggests an initial attempt to use a different,
  // possibly non-standard, 'contains' function.
  //	if(contains(s,t)||contains(s,o))

  // Block Logic: Check if the input string `s` contains either of the
  // dangerous substrings.
  if (strstr(s.c_str(), t.c_str()) || strstr(s.c_str(), o.c_str()))
    cout << "YES";
  else
    cout << "NO";
}
