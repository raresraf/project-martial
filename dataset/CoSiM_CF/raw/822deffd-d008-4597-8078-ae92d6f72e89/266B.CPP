#include <iostream>
#include <string>

int main() {
  std::ios::sync_with_stdio(false);
  std::string order;
  int n, t;
  std::cin >> n >> t;
  std::cin >> order;

  char tmp;
  while (t--) {
    int i = 0;
    while (i < n - 1) {
      while (order[i] == 'G')
        ++i;
      while (order[i + 1] == 'B')
        ++i;
      if (i < n - 1) {
        char tmp = order[i];
        order[i] = order[i + 1];
        order[i + 1] = tmp;
        i += 2;
      }
    }
  }
  std::cout << order << std::endl;

  return 0;
}