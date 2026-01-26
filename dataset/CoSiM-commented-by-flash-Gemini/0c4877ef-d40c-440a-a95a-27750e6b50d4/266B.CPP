#include <iostream>
#include <string>

int main() {
  int n, t;
  std::string queue;

  std::cin >> n >> t;

  std::cin >> queue;

  for (int j = 0; j < t; j++) {
    for (int i = 0; i < n - 1; i++) {
      // if boy is before girl
      if ('B' == queue[i] && 'G' == queue[i + 1]) {
        // change positions
        queue[i] = 'G';
        queue[i + 1] = 'B';
        i++;
      }
    }
  }

  std::cout << queue << std::endl;

  return 0;
}
