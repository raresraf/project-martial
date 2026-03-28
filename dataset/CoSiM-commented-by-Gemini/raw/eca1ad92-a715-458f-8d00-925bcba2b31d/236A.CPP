/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their name.
 * 
 * This program reads a username and determines whether to "chat with her" or "ignore him".
 * The decision is based on whether the number of distinct characters in the username is
 * even or odd.
 * 
 * @b Algorithm: The number of distinct characters is found by sorting the input string
 *    and then using the erase-unique idiom to remove duplicate characters. The parity
 *    of the length of the resulting string is then checked.
 * @b Time_Complexity: O(L log L), where L is the length of the string, dominated by the
 *    sorting operation.
 * @b Space_Complexity: O(L) to store the string. The operations are performed in-place.
 */
// 4022081   Jul 4, 2013 8:27:39 PM	fuwutu	 236A - Boy or Girl	 GNU
// C++0x	Accepted	15 ms	0 KB
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  // Sort the string to bring all identical characters together.
  sort(s.begin(), s.end());

  /**
   * @brief Use the erase-unique idiom to remove duplicate characters.
   * `unique` shunts all unique elements to the beginning of the range and returns
   * an iterator to the end of the unique range. `erase` then removes the
   * leftover duplicate elements from the end of the string.
   */
  s.erase(unique(s.begin(), s.end()), s.end());

  // The decision is based on the parity of the count of distinct characters.
  if (s.length() % 2 == 0) {
    // If the count of distinct characters is even.
    cout << "CHAT WITH HER!" << endl;
  } else {
    // If the count of distinct characters is odd.
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}