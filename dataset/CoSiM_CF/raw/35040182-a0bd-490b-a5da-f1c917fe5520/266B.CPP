#include <iostream>

int main() {
  int n, t;
  std::string queue;
  std::cin >> n >> t;
  std::cin >> queue;

  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n; j++) {
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        std::swap(queue[j], queue[j + 1]);
        j++;
      }
    }
  }

  std::cout << queue;

  return 0;
}