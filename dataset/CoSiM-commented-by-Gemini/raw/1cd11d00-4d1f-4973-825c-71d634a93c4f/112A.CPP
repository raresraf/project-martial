/**
 * @file 112A.CPP
 * @brief Solution for Codeforces problem 112A, "Petya and Strings".
 *
 * @description
 * This program performs a case-insensitive lexicographical comparison of two
 * input strings. It outputs -1 if the first string is lexicographically smaller
 * than the second, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm
 * This implementation uses a "convert-then-compare" strategy.
 * 1. Read two strings, s and s2.
 * 2. Iterate through both strings, converting all uppercase characters to lowercase.
 * 3. After the conversion, the C++ string comparison operators (`<`, `>`, `==`)
 *    are used to perform a lexicographical comparison on the now fully
 *    lowercase strings.
 * 4. The result (-1, 1, or 0) is stored and then printed.
 *
 * @time_complexity O(L), where L is the length of the strings. The entire
 *   length of the strings is traversed for case conversion, and then again for
 *   the final comparison.
 * @space_complexity O(L) to store the input strings.
 */
#include <iostream>
using namespace std;
int main() {
  string s, s2;
  cin >> s >> s2;

  // Block Logic: Convert both strings to lowercase in-place.
  for (int i = 0; i < s.length(); i++) {
    // Check for uppercase ASCII values (A=65, Z=90).
    if (s[i] >= 65 and s[i] <= 90) {
      // Add 32 to convert to the corresponding lowercase letter's ASCII value.
      s[i] += 32;
    }
    if (s2[i] >= 65 and s2[i] <= 90) {
      s2[i] += 32;
    }
  }

  // The commented-out line below was likely for debugging the conversion.
  //	cout<<s<<endl<<s2<<endl;
  int d = 0;
  // This commented-out block shows an alternative but incomplete comparison logic.
  /*for(int i=0;i<s.length();i++)
  {
          d=s[i]-s2[i];
  }*/

  // Block Logic: Perform lexicographical comparison on the normalized (lowercase) strings.
  if (s < s2) {
    d = -1;
  } else if (s > s2) {
    d = 1;
  } else if (s == s2) {
    d = 0;
  }
  cout << d;
}
