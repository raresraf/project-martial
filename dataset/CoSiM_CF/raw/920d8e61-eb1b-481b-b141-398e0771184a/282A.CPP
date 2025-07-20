#include <iostream>

using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  char a[n][3];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    if (a[i][1] == '+') {
      x++;
    }
    if (a[i][1] == '-')
      x--;
  }
  cout << x;
  return 0;
}
