/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of the input string.
 *
 * This program reads a string from standard input and determines whether
 * the word "hello" can be obtained by deleting zero or more characters
 * from the input string. It's a classic subsequence problem.
 *
 * @algorithm Greedy Subsequence Search
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string. This is because
 * each call to `find_first_of` scans a part of the string, and in total, the
 * entire string is scanned at most a constant number of times.
 * Space Complexity: O(1), for storing the indices of the found characters,
 * excluding the space for the input string itself.
 */

#include <iostream>
#include <string>
using namespace std;
int main() {
  string x;
  int i = 0, arr[5];
  cin >> x;

  /**
   * @brief Sequentially find the characters of "hello" in the input string.
   *
   * The logic greedily searches for each character of "hello" after the position
   * of the previously found character. This ensures that the subsequence order
   * is maintained.
   */
  // Find the first 'h'
  i = x.find_first_of('h', i);
  arr[0] = i;
  // Find the first 'e' after 'h'
  i = x.find_first_of('e', i);
  arr[1] = i;
  // Find the first 'l' after 'e'
  i = x.find_first_of('l', i);
  arr[2] = i;
  // Find the second 'l' after the first 'l'
  i = x.find_first_of('l', i);
  if (i != arr[2]) {
    arr[3] = i;
  } else {
    // If the same 'l' is found, search for the next one.
    i = x.find_first_of('l', i + 1);
    arr[3] = i;
  }
  // Find the first 'o' after the second 'l'
  i = x.find_first_of('o', i);
  arr[4] = i;

  /**
   * @brief Validate if all characters were found.
   *
   * If any character was not found, its position in `arr` will be -1 (string::npos).
   * If any position is -1, it means "hello" cannot be formed as a subsequence.
   */
  for (i = 0; i < 5; i++) {
    if (arr[i] == -1) {
      cout << "NO";
      return 0;
    }
  }

  // If all characters are found in the correct order, "hello" is a subsequence.
  cout << "YES";
  return 0;
}
