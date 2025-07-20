#include <iostream>

using namespace std;

int main() {
  int m, n, area, counter = 0;

  cin >> m >> n;
  area = m * n;

  while (area >= 2) {
    counter++;
    area -= 2;
  }

  cout << counter;
  return 0;
}