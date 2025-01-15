#include <iostream>
#include <regex>
#include <string>

bool implement(int a, int b, int c) {
  if (a && b || a && c || b && c) {
    return true;
  } else {
    return false;
  }
}

int main() {
  int n, a, b, c;

  int implementCounter = 0;

  std::cin >> n;

  for (int i = 0; i < n; i++) {
    std::cin >> a >> b >> c;

    if (implement(a, b, c)) {
      implementCounter++;
    }
  }

  std::cout << implementCounter << std::endl;

  return 0;
}
