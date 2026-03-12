/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 *
 * This program determines a user's gender based on the number of distinct
 * characters in their username. If the count of distinct characters is even,
 * the user is identified as female ("CHAT WITH HER!"). If the count is odd,
 * the user is identified as male ("IGNORE HIM!").
 *
 * @algorithm Set-based character counting.
 * @time_complexity O(N log K), where N is the length of the username and K is the number of distinct characters (at most 26). The log factor comes from inserting into the std::set.
 * @space_complexity O(K) to store the distinct characters in the set.
 */
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  set<char> s;
  char c;

  /**
   * @brief Reads the username and stores its distinct characters in a set.
   *
   * The loop reads characters one by one until a newline is encountered.
   * Using a std::set automatically handles the storage of only unique characters.
   */
  while (true) {
    scanf("%c", &c);
    if (c == '\n')
      break;

    s.insert(c);
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the size of the set (number of distinct characters) is even, the output
   * is "CHAT WITH HER!". Otherwise, it is "IGNORE HIM!".
   */
  if (s.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}