#include <iostream>

int main() {
  std::string s1, s2;
  std::cin >> s1 >> s2;
  int temp1, temp2;
  for (unsigned int i = 0; i < s1.size(); ++i) {
    temp1 = tolower(s1[i]), temp2 = tolower(s2[i]);
    if (temp1 < temp2) {
      std::cout << "-1\n";
      return 0;
    } else if (temp1 > temp2) {
      std::cout << "1\n";
      return 0;
    }
  }
  std::cout << "0\n";
}
