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

#include <iostream>
#include <string>

int main() {
  std::string inputStr, hello, outputStr;
  unsigned int i, j;
  hello = "hello";
  outputStr = "NO";
  std::cin >> inputStr;

  for (i = j = 0; i < inputStr.length(); i++) {
    if (inputStr.at(i) == hello.at(j)) {
      if (++j == 5) {
        outputStr = "YES";
        break;
      }
    }
  }

  std::cout << outputStr << '\n';
  return 0;
}
