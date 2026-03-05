/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem using a hash set.
 *
 * This program determines if a user is male or female based on their username.
 * The rule is: if the number of distinct characters in the username is even,
 * the user is female ("CHAT WITH HER!"). If the count is odd, the user is male
 * ("IGNORE HIM!").
 *
 * @algorithm
 * This implementation uses a `std::unordered_set` to efficiently count the
 * number of distinct characters. A set is a data structure that stores only
 * unique elements.
 *
 * 1. Read the input username into a string `a`.
 * 2. Create an empty `unordered_set` of characters named `b`.
 * 3. Iterate through each character `c` in the input string `a`.
 * 4. For each character, insert it into the set `b`. The set automatically
 *    handles duplicates; if a character is already present, the set remains
 *    unchanged.
 * 5. After the loop, the size of the set, `b.size()`, will be equal to the
 *    number of distinct characters in the username.
 * 6. Check if `b.size()` is even by using the modulo operator (`b.size() % 2 == 0`).
 *    a. If true (the count is even), print "CHAT WITH HER!".
 *    b. If false (the count is odd), print "IGNORE HIM!".
 *
 * This is arguably the most direct and idiomatic C++ approach to solving this problem.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string. On average, insertion
 * into an `unordered_set` is an O(1) operation.
 * Space Complexity: O(K), where K is the number of distinct characters (at most 26
 * for lowercase English letters).
 */

#include <iostream>
#include <string>
#include <unordered_set>
// Other includes are not strictly necessary for this specific code.
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

int main() {
  // `a` stores the input username.
  string a;
  // `b` is a hash set that will only store the unique characters from the username.
  unordered_set<char> b;
  cin >> a;

  /**
   * @brief Populate the set with characters from the username.
   * Duplicates are automatically ignored by the set.
   */
  for (char c : a) {
    b.insert(c);
  }

  // After the loop, `b.size()` gives the count of distinct characters.
  // Check if the count of distinct characters is even.
  if (b.size() % 2 == 0) {
    // If even, the user is female.
    cout << "CHAT WITH HER!";
  } else {
    // If odd, the user is male.
    cout << "IGNORE HIM!";
  }
  
  // return 0 is implicit at the end of main.
}
