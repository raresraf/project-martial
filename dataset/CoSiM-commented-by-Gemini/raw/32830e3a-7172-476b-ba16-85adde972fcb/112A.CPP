/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem by case-normalization.
 * @details This program compares two input strings lexicographically, ignoring case.
 * It achieves this by first converting both strings to lowercase and then using
 * standard string comparison operators.
 *
 * @algorithm The core of the algorithm is normalization before comparison.
 * 1. A helper function `lower` is defined to convert a string to lowercase in-place.
 * 2. Two strings are read from input.
 * 3. The `lower` function is called on both strings.
 * 4. The now all-lowercase strings are compared using `==` and `<` operators to
 *    determine if they are equal, or which one is lexicographically smaller.
 * 5. The result (0, -1, or 1) is printed accordingly.
 *
 * @time_complexity O(L), where L is the length of the strings. The complexity is
 * dominated by the conversion to lowercase and the final comparison, both of which
 * are linear operations.
 * @space_complexity O(L), to store the two input strings.
 */
#include <algorithm>
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Converts all uppercase characters in a string to lowercase, in-place.
 * @param s A reference to the string to be modified.
 */
void lower(string &s) {
  int n = s.length();
  for (int i = 0; i < n; i++) {
    // cout<<isupper(s[i]);
    if (isupper(s[i])) {
      s[i] = tolower(s[i]);
      // cout<<"s[i]="<<s[i]<<endl;
    }
  }
  // cout<<"After:"<<s<<endl;
}

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  // Block Logic: Normalize both strings to lowercase before comparison.
  lower(s1);
  lower(s2);
  // cout<<s1<<endl;
  //  cout<<s2<<endl;
  /**
   * Block Logic: Perform lexicographical comparison on the normalized strings.
   */
  if (s1 == s2)
    cout << 0;
  else if (s1 < s2)
    cout << -1;
  else
    cout << 1;

  return 0;
}
