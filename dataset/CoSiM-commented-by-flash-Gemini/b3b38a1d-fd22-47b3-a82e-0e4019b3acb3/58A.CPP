/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence of a given input string.
 * Algorithm: Iterative search using string::find_first_of to locate characters sequentially.
 * Time Complexity: O(L * K) where L is the length of the input string and K is the length of the subsequence ("hello" is constant, so O(L)).
 * Space Complexity: O(1) as only a few variables and a fixed-size array are used.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string x;
  int i = 0, arr[5];
  cin >> x;

  /**
   * @brief Sequentially searches for each character of "hello" in the input string 'x'.
   * Pre-condition: 'x' is the input string. 'i' is the starting search index (initially 0),
   * and 'arr' stores the found indices of "h", "e", "l", "l", "o".
   * Invariant: Each search for a character starts from the index after the previously found character,
   * ensuring that the subsequence is found in order.
   */
  i = x.find_first_of('h', i);
  arr[0] = i;
  i = x.find_first_of('e', i);
  arr[1] = i;
  i = x.find_first_of('l', i);
  arr[2] = i;
  i = x.find_first_of('l', i);
  /**
   * @brief Handles the case for finding the second 'l' in "hello".
   * Logic: If the index of the second 'l' is the same as the first 'l', it means
   * `find_first_of` found the same character again. In this case, it searches for the
   * second 'l' starting from the position *after* the first 'l'.
   */
  if (i != arr[2]) {
    arr[3] = i;
  } else {
    i = x.find_first_of('l', i + 1);
    arr[3] = i;
  }
  i = x.find_first_of('o', i);
  arr[4] = i;
  /**
   * @brief Validates if all characters of "hello" were found.
   * Pre-condition: 'arr' contains the indices found for each character of "hello".
   * Invariant: If any character was not found, its corresponding index in 'arr' will be -1.
   */
  for (i = 0; i < 5; i++) {
    if (arr[i] == -1) {
      cout << "NO";
      return 0;
    }
  }
  // All characters of "hello" were found as a subsequence.
  cout << "YES";
  return 0;
}
