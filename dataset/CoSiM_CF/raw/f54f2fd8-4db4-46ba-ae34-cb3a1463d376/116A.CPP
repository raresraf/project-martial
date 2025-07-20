#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int s = 0;
  int c = 0;
  int a[n][2];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    s = s - a[i][0] + a[i][1];
    c = max(c, s);
  }
  cout << c;
  return 0;
}
