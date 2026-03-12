/*
58A
Chat room
Vasya has recently learned to type and log on to the Internet. He immediately
entered a chat room and decided to say hello to everybody. Vasya typed the word
s. It is considered that Vasya managed to say hello if several letters can be
deleted from the typed word so that it resulted in the word "hello". For
example, if Vasya types the word "ahhellllloou", it will be considered that he
said hello, and if he types "hlelo", it will be considered that Vasya got
misunderstood and he didn't manage to say hello. Determine whether Vasya managed
to say hello by the given word s.

Input
The first and only line contains the word s, which Vasya typed. This word
consisits of small Latin letters, its length is no less that 1 and no more than
100 letters.

Output
If Vasya managed to say hello, print "YES", otherwise print "NO".
*/
/* Accepted 15 ms 2152 KB */

/**
 * @file 58A.CPP
 * @brief Determines if the string "hello" is a subsequence of a given input string.
 *
 * This program solves the "Chat room" problem where the goal is to check if
 * the word "hello" can be formed by deleting zero or more characters from an
 * input string. This is equivalent to checking for a subsequence.
 *
 * @algorithm Greedy Subsequence Matching
 * @time_complexity O(N), where N is the length of the input string, as it involves a single pass.
 * @space_complexity O(N) to store the input string.
 */
#include <iostream>
#include <string>

int main() {
  std::string inputStr, hello, outputStr;
  unsigned int i, j;
  hello = "hello";
  outputStr = "NO";
  std::cin >> inputStr;

  /**
   * @brief Scans the input string to find "hello" as a subsequence.
   *
   * This loop iterates through the input string `inputStr` with index `i`, while
   * using index `j` to track the current character being sought from the target
   * string "hello". It's a greedy approach that finds the first character,
   * then searches for the next character from that point onward.
   */
  for (i = j = 0; i < inputStr.length(); i++) {
    // If the current character in the input matches the target character in "hello"...
    if (inputStr.at(i) == hello.at(j)) {
      // ...move to the next character in "hello".
      if (++j == 5) { // If all 5 characters of "hello" have been found...
        outputStr = "YES";
        break; // ...the subsequence is found, so we can exit the loop.
      }
    }
  }

  std::cout << outputStr << '\n';
  return 0;
}