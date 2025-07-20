#include <iostream>

using namespace std;

int main() {
  int q, c, m, x, teams;
  cin >> q;
  for (int i = 0; i < q; i++) {
    cin >> c >> m >> x;
    teams = 0;
    while (c > 0 && m > 0) {
      c--;
      m--;
      if (x > 0) {
        x--;
        teams++;
      } else {
        if (c > 0 && c >= m) {
          c--;
          teams++;
        } else if (m > 0 && m > c) {
          m--;
          teams++;
        }
      }
    }

    cout << teams << endl;
  }

  return 0;
}