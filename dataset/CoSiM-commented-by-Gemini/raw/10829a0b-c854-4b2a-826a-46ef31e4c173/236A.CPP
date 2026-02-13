/**
 * @file 236A.CPP
 * @brief An attempted solution for the "Boy or Girl" problem (Codeforces 236A).
 * @author competitive-programmer
 *
 * @details This program attempts to solve the "Boy or Girl" problem, which involves
 * counting the number of distinct characters in a username to determine a gender.
 * If the count is even, the output should be "CHAT WITH HER!"; if odd, "IGNORE HIM!".
 *
 * @warning This implementation contains several logical errors. It incorrectly
 * counts the number of distinct characters and the final decision logic is
 * reversed. It should not be used as a reference solution.
 *
 * Flawed Algorithm:
 * 1. Read the input string `a`.
 * 2. Initialize a counter `pas` to 1.
 * 3. Use a nested loop to check for distinct characters. For each character `a[i]`,
 *    it checks if it has appeared before in `a[0...i-1]`.
 * 4. If no duplicate is found in the prefix, `pas` is incremented.
 * 5. Finally, it checks the parity of `pas` to make a decision.
 */
#include <iostream>
using namespace std;

string a;
// `pas` is intended to count distinct characters, but is initialized incorrectly.
int pas = 1, num;

int main() {
  cin >> a;
  /**
   * Block Logic: This nested loop attempts to count distinct characters.
   * The outer loop iterates through each character. The inner loop checks if
   * the current character has appeared before.
   */
  for (int i = 0; i < a.size(); i++) {
    // `num` is used as a flag to check for duplicates for the character a[i].
    for (int j = 0; j < i; j++) {
      if (a[i] == a[j]) {
        num++;
        // Note: The inner loop is inefficient as it doesn't break after finding
        // the first duplicate.
      }
    }
    
    // If `num` is 0, it means a[i] is the first occurrence of this character.
    if (num == 0) {
      // @bug Logical Error: The counter `pas` is initialized to 1, which makes the
      // final count of distinct characters incorrect. For an input "a", the
      // final `pas` will be 2, not 1.
      pas++;
    }
    num = 0; // Reset duplicate flag for the next character.
  }
  
  /**
   * @bug Logical Error: The decision logic is reversed. According to the problem,
   * an even number of distinct characters should result in "CHAT WITH HER!".
   * This code does the opposite.
   */
  if (pas % 2 == 0) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
}
