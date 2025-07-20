#include <iostream>

using namespace std;

int main() {
  int a;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a;
      if (a == 1) {
        int count = abs(3 - i) + abs(3 - j);
        printf("%d", count);
        break;
      }
    }
  }
}