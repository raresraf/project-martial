/**
 * @file 236A_2.CPP
 * @brief Solution for the "Boy or Girl" problem (e.g., Codeforces 236A).
 * @details This program determines if the number of distinct characters in a username
 * is even or odd. If even, it prints "CHAT WITH HER!"; if odd, it prints "IGNORE HIM!".
 * This implementation uses the `std::sort` and `std::unique` algorithms.
 *
 * @algorithm Sort and find unique elements.
 * @complexity
 * - Time: O(L log L), where L is the length of the string, dominated by the `std::sort` operation.
 * - Space: O(L) to store the characters of the string in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;

  // A vector to hold the characters for sorting.
  vector<char> v;
  for (int i = 0; i < s.size(); i++) {
    v.push_back(s[i]);
  }

  /*  A commented-out alternative for initializing the vector from a C-style string.
      OR,  char s[100];
           cin >> s;
           len = strlen(s);
           vector<char>v(s,s+len);
  */

  // Sort the vector. This places all identical characters adjacent to each other.
  sort(v.begin(), v.end());

  // `std::unique` rearranges the elements in the sorted vector so that each
  // unique element appears once at the beginning of the range.
  // It returns an iterator to the new "logical" end of the unique range.
  vector<char>::iterator it;
  it = unique(v.begin(), v.end());

  // Calculate the number of unique elements by finding the distance from the
  // beginning of the vector to the iterator returned by `unique`.
  int curent_length = distance(v.begin(), it);

  // Check if the count of distinct characters is even or odd.
  if (curent_length % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }

  // --- Debugging Output ---
  // The following lines are for debugging and would typically be removed in a final submission.
  cout << "
" << distance(v.begin(), it) << endl;

  for (int i = 0; i < s.size(); i++) {
    cout << v[i] << " ";
  }
  // Note: The vector `v` has been modified in-place by `std::unique`,
  // so this output will show the unique elements followed by unspecified values.

  return 0;
}
