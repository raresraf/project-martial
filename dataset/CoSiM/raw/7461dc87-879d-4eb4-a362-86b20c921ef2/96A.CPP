#include <iostream>

int main() {
  int nb0 = 0, nb1 = 0;
  ;
  std::string s;
  std::cin >> s;
  for (int i = 0; i <= s.size(); i++) {
    if (s[i] == '1') {
      nb1++;
      nb0 = 0;
    }
    if (s[i] == '0') {
      nb1 = 0;
      nb0++;
    }
    if (nb0 >= 7 or nb1 >= 7) {
      std::cout << "YES";
      return 0;
    }
  }
  std::cout << "NO";
}
