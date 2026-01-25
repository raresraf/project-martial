#include <iostream>

int main() {
  std::string word;
  std::string result;
  std::cin >> word;
  char c;

  for (int i = 0; i < word.length(); i++) {
    c = tolower(word[i]);
    switch (c) {
    case 'a':
    case 'i':
    case 'u':
    case 'e':
    case 'o':
    case 'y':
      continue;
    }
    result.append(1, '.');
    result.append(1, c);
  }
  std::cout << result << std::endl;
  return 0;
}