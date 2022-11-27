#include <iostream>

std::string evaluate(int &in) {
  if (in >= 4 && in % 2 == 0)
    return "YES";
  else
    return "NO";
}

int main() {
  int in = 0;
  std::cin >> in;
  std::cout << evaluate(in);
  return 0;
}
