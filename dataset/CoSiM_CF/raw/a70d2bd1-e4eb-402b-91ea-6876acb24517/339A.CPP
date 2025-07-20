#include <bits/stdc++.h>

int main() {
  std::string s;
  std::cin >> s;
  unsigned n1 = 0, n2 = 0, n3 = 0;
  for (int i = 0; i < s.size(); i += 2)
    if (s[i] == '1')
      n1++;
    else if (s[i] == '2')
      n2++;
    else
      n3++;
  if (n1 > 0) {
    std::cout << "1";
    n1--;
  } else if (n2 > 0) {
    std::cout << "2";
    n2--;
  } else if (n3 > 0) {
    std::cout << "3";
    n3--;
  }
  while (n1--) {
    std::cout << "+1";
  }
  while (n2--) {
    std::cout << "+2";
  }
  while (n3--) {
    std::cout << "+3";
  }
  std::cout << std::endl;
}
