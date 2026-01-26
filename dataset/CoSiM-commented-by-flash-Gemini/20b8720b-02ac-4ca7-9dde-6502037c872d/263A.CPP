#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int a, b, n, total = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n;
      if (n == 1) {
        total = abs(i - 2) + abs(j - 2);
      }
    }
  }
  cout << total;
  return 0;
}
