/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 * @details This program abbreviates words longer than 10 characters. It contains a bug in its abbreviation logic.
 *
 * Algorithm:
 * 1. Read the number of test cases, `test`.
 * 2. Loop `test` times.
 * 3. For each test case, read a single word `testcase`.
 * 4. If the word's length is greater than 10, it should be abbreviated.
 * 5. The intended abbreviation is to print the first character, the number of characters
 *    between the first and last (i.e., `length - 2`), and the last character.
 * 6. If the word's length is 10 or less, it is printed unchanged.
 *
 * @warning Buggy Implementation: The code calculates the middle number as `(length - 2) % 100`.
 * The modulo 100 operation is incorrect. For a string of length 112, the length of the middle part is 110,
 * but `110 % 100` is 10. The output would be incorrect (e.g., "t10r" instead of "t110r").
 * This bug might not be triggered if the problem constraints keep the maximum string length below 112.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the max length of a word.
 * Space Complexity: O(L) for storing the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int test;
  cin >> test;

  while (test--) {
    string testcase;
    cin >> testcase;
    int length = testcase.length();
    
    /**
     * Block Logic: Check if the word is "too long" and abbreviate it.
     */
    if (length > 10) {
      // The logic to calculate the number of middle characters is buggy due to the unnecessary modulo 100.
      cout << testcase[0] << (length - 2) % 100 << testcase[length - 1] << endl;
    } else {
      cout << testcase << endl;
    }
  }
  return 0;
}
