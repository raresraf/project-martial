/**
 * @file 236A.CPP
 * @brief Determines the user's gender based on the number of distinct characters in their username.
 *
 * This program reads a username as a string and counts the number of unique characters.
 * If the count of unique characters is even, it prints "CHAT WITH HER!".
 * Otherwise (if the count is odd), it prints "IGNORE HIM!".
 *
 * @algorithm Character Counting using a Set
 * @details The program utilizes `std::set` to automatically handle the counting of
 * unique characters due to the properties of a set data structure.
 *
 * @complexity
 * Time Complexity: O(N * log K), where N is the length of the input string and K is the
 * number of unique characters. For each character in the string, insertion into the set
 * takes logarithmic time in the size of the set. Since the number of unique characters (K)
 * is small and bounded by the size of the character set (e.g., ASCII), this can be
 * considered approximately O(N).
 * Space Complexity: O(K), where K is the number of unique characters, to store them in the set.
 * In the worst case, this is proportional to the size of the character set.
 */

#include <iostream>
#include <set>
#include <string>

using namespace std;

int main() {
  string str;
  set<char> c;

  cin >> str;

  /**
   * @brief Insert all characters from the string into a set to find unique characters.
   * The std::set will only store unique elements, so the final size of the set 'c'
   * will be the number of distinct characters in the string 'str'.
   */
  for (int i = 0; i < str.length(); ++i) {
    c.insert(str[i]);
  }

  /**
   * @brief Check if the number of distinct characters is even or odd.
   * The problem statement implies a decision based on the parity of the
   * count of unique characters.
   */
  if (c.size() % 2 == 0) {
    // If the count is even, the user is determined to be female.
    cout << "CHAT WITH HER!" << endl;
    return 0;
  }

  // If the count is odd, the user is determined to be male.
  cout << "IGNORE HIM!" << endl;

  return 0;
}
