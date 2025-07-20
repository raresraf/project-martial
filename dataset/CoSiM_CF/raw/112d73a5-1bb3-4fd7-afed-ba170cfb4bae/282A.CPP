#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int n;
  std::cin >> n;

  std::string command;
  char op;
  int count = 0;
  while (n--) {
    std::cin >> command;
    op = command[1];
    if (op == '+') {
      count++;
    } else if (op == '-') {
      count--;
    }
  }
  std::cout << count << std::endl;
  return 0;
}