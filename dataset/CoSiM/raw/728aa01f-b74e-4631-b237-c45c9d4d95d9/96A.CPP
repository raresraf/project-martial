#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::string str;
  std::cin >> str;

  bool dangerous = false;
  int conseq = 1;
  for (size_t i = 1; i < str.size(); ++i) {
    if (str[i - 1] != str[i]) {
      conseq = 1;
      continue;
    }

    conseq++;
    if (conseq == 7) {
      dangerous = true;
      break;
    }
  }

  std::cout << (dangerous ? "YES" : "NO") << std::endl;
}
