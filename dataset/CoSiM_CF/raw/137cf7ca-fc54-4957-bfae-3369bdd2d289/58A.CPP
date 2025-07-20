#include <iostream>

std::string find_next_char_from_word(std::string str, std::string word) {
  int pos = str.find(word[0]);
  if (word.compare("") == 0) {
    return "YES";
  }
  if (pos == std::string::npos) {
    return "NO";
  } else {
    std::string new_word = word.substr(1, word.length() - 1);
    std::string new_str = str.substr(pos + 1, str.length() - pos - 1);
    return find_next_char_from_word(new_str, new_word);
  }
}

int main() {
  std::ios::sync_with_stdio(false);
  std::string str;
  std::cin >> str;

  std::string word = "hello";
  std::cout << find_next_char_from_word(str, word) << std::endl;
  return 0;
}