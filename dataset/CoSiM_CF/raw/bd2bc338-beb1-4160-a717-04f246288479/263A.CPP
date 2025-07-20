#include <cmath>
#include <iostream>
#include <string>

using namespace std;

int main() {
  int a[6][6] = {};
  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++) {
      cin >> a[i][j];
      if (a[i][j]) {
        cout << abs(i - 3) + abs(j - 3);
        return 0;
      }
    }
}