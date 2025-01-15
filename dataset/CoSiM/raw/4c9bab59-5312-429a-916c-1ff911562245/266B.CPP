#include <iostream>

void changePositions(char *values, int n, int t) {
  while (true) {
    if (t == 0)
      return;
    for (int i = 0; i < n - 1; ++i) {
      if ((values[i] == 'B') && (values[i + 1] == 'G')) {
        values[i] = 'G';
        values[i + 1] = 'B';
        ++i;
      }
    }
    --t;
  }
}

int main() {
  int n, t;
  std::cin >> n >> t;
  char *values = new char[n];
  for (int i = 0; i < n; ++i) {
    std::cin >> values[i];
  }
  changePositions(values, n, t);
  for (int i = 0; i < n; ++i) {
    std::cout << values[i];
  }
  std::cout << '\n';
}
