/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem by constructing an abbreviated string.
 * @details For each input word, if its length is greater than 10, this program constructs
 * a new string for the abbreviation. It combines the first character, the length of the
 * middle part as a string, and the last character. This new string is then printed.
 * Otherwise, the original word is printed.
 *
 * @author Nikhil Nagrale (nikhilnagrale2)
 *
 * @algorithm String construction and conditional printing.
 * @complexity
 * Time complexity: O(N * L), where N is the number of words and L is the maximum length. The string construction inside the loop adds a small constant overhead compared to direct printing.
 * Space complexity: O(L) to store the input word and potentially the new abbreviated string.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      // Create a new string to build the abbreviation.
      string ans;
      // Add the first character.
      ans.push_back(s.front());
      // Convert the length of the middle part to a string and append it.
      ans += to_string(s.length() - 2);
      // Add the last character.
      ans += s.back();
      cout << ans << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}
