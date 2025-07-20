#include <iostream>

int main() {
  int n;
  std::cin >> n;
  int count = 0;
  while (n--) {
    int temp1, temp2, temp3;
    std::cin >> temp1 >> temp2 >> temp3;
    count += ((temp1 && temp2) || (temp2 & temp3) || (temp3 && temp1));
  }
  std::cout << count << '\n';
}
