/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem.
 * @contest Code Forces Round #146 (Div. 2)
 * @url http://codeforces.ru/contest/236/problem/A
 *
 * @problem The program determines a user's gender based on their username. If the
 * number of distinct characters in the username is odd, the user is male ("IGNORE HIM!").
 * If the number is even, the user is female ("CHAT WITH HER!").
 *
 * @algorithm The algorithm uses a `std::set` to efficiently count the number of
 * unique characters in the input string.
 * 1. Read the username string.
 * 2. Iterate through the string and insert each character into the `std::set`.
 *    The set automatically handles uniqueness, so duplicates are not stored.
 * 3. After the loop, the size of the set is equal to the number of distinct characters.
 * 4. Check if the size is odd or even using a bitwise AND operation (`size & 1`).
 * 5. Print the corresponding message based on the parity.
 *
 * @time_complexity O(N log K), where N is the length of the string and K is the
 * number of unique characters. Each insertion into the set takes O(log K) time.
 * @space_complexity O(K) to store the unique characters in the set.
 */
#include <iostream>
#include <set>
#include <string>

int main(int argc, char **argv) {
  // Use a std::set to store unique characters from the input string.
  std::set<int> ch;

  std::string s;
  std::cin >> s;

  // Insert each character of the string into the set.
  // The set will only store one copy of each unique character.
  for (auto it = s.cbegin(); it != s.cend(); ++it) {
    ch.insert(*it);
  }

  // Check the parity of the number of distinct characters.
  // (ch.size() & 1) is a bitwise trick to check if a number is odd.
  // If the size is odd, the result is 1 (true); if even, it's 0 (false).
  std::cout << ((ch.size() & 1) ? "IGNORE HIM!" : "CHAT WITH HER!")
            << std::endl;
  return 0;
}
