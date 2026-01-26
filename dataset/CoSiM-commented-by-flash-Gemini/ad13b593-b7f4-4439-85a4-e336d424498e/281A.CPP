/*
281A
Word Capitalization
Capitalization is writing a word with its first letter as a capital letter. Your
task is to capitalize the given word.

Note, that during capitalization all the letters except the first one remains
unchanged.

Input
A single line contains a non-empty word. This word consists of lowercase and
uppercase English letters. The length of the word will not exceed 103.

Output
Output the given word after capitalization.
*/
/* Accepted 60 ms 2144 KB */

#include <ctype.h>
#include <iostream>
#include <string>

int main() {
  std::string word;
  std::cin >> word;
  word.at(0) = toupper(word.at(0));
  std::cout << word << '\n';
  return 0;
}
