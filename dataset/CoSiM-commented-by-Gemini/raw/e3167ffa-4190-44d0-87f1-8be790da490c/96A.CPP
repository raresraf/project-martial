/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 * @description This program determines if a situation is "dangerous" by checking if a
 *              given string of '0's and '1's contains a sequence of at least seven
 *              identical consecutive characters.
 *
 * Algorithm: This solution uses the `string::find` method from the C++ standard library.
 * 1. It defines two target substrings: "0000000" and "1111111".
 * 2. It reads the input string.
 * 3. As an optimization, it first checks if the input string's length is less than 7.
 *    If it is, a dangerous situation is impossible, and it prints "NO".
 * 4. Otherwise, it searches for the occurrence of either of the two target substrings
 *    within the input string.
 * 5. If `string::find` returns a valid position for either substring, a dangerous
 *    situation exists, and the program prints "YES". Otherwise, it prints "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string. The `find` operation
 *                  is linear in time complexity.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  std::string str;
  cin >> str;
  // Define the two "dangerous" sequences.
  std::string str0(7, '0');
  std::string str1(7, '1');

  // Pre-condition: A dangerous situation is only possible if the string is at least 7 characters long.
  if (str.length() < 7) {
    std::cout << "NO";
  } else {
    // Search for the occurrence of seven consecutive '0's.
    std::size_t found0 = str.find(str0);
    // Search for the occurrence of seven consecutive '1's.
    std::size_t found1 = str.find(str1);
    
    // `string::npos` is returned by `find` if the substring is not found.
    if (found0 == std::string::npos && found1 == std::string::npos) {
      // If neither sequence was found, the situation is not dangerous.
      std::cout << "NO";
    } else {
      // If either sequence was found, the situation is dangerous.
      std::cout << "YES";
    }
  }

  return 0;
}
