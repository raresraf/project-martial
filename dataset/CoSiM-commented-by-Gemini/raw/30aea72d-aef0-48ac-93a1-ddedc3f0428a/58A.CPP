/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem by checking for a subsequence.
 * @details This program determines if the word "hello" can be formed by deleting zero or
 * more characters from a given input string. In other words, it checks if "hello" is a
 * subsequence of the input string.
 *
 * @algorithm The solution uses a greedy two-pointer approach. One pointer (`i`) iterates
 * through the target string "hello", while a second pointer (`inc`) traverses the input
 * string. The algorithm scans the input string to find the first character 'h'. Once found,
 * it continues scanning from the next position to find 'e', and so on. If all five
 * characters of "hello" are found in the correct order, it's a valid subsequence.
 *
 * @time_complexity O(N), where N is the length of the input string. Despite the nested
 * loops, the pointer for the input string (`inc`) is never reset, ensuring that each
 * character in the input is processed at most once.
 * @space_complexity O(1), as it uses a fixed amount of extra space regardless of the input size.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // counter: Tracks how many characters of "hello" have been found so far.
  int counter = 0;
  // inc: A persistent index (pointer) for the input string, not reset in the outer loop.
  int inc = 0;
  char hello[5] = {'h', 'e', 'l', 'l', 'o'};
  string input;
  cin >> input;
  /**
   * Block Logic: This structure implements a two-pointer scan to find the "hello" subsequence.
   * The outer loop iterates through the target `hello` array ('i' is the first pointer).
   * The inner loop scans the `input` string from where the last character was found
   * ('inc' is the second pointer).
   */
  for (int i = 0; i < 5; i++) {
    for (inc; inc < input.length(); inc++) {
      // Pre-condition: Looking for the character `hello[i]` in the remainder of the input string.
      if (hello[i] == input.at(inc)) {
        counter++;
        // Move the input string pointer to the next position for the subsequent search.
        inc++;
        break;
      }
    }
  }
  // Block Logic: Check if all five characters of "hello" were found in order.
  if (counter == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
