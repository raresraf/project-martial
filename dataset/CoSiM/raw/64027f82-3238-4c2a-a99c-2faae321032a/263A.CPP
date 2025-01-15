#include <algorithm>
#include <iostream>
using namespace std;

int main() {

  int t = 0, i, j;

  for (i = 1; i <= 5 && t != 1; i++) {
    for (j = 1; j <= 5 && t != 1; j++) {
      cin >> t;
    }
  }
  i--;
  j--;
  cout << abs(3 - i) + abs(3 - j);
  return 0;
}
