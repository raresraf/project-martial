#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int m[n][3];

  int ans = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> m[i][j];
    }
  }

  for (int i = 0; i < n; i++) {

    int y = 0;
    for (int j = 0; j < 3; j++) {

      y = y + m[i][j];
    }

    if (y >= 2) {
      ans = ans + 1;
    }
  }

  cout << "" << ans;

  return 0;
}
