#include <bits/stdc++.h>
using namespace std;
int main() {

  int tr[2], c = 0, d = 0, test;
  cin >> test;
  for (int i = 0; i < test; i++) {
    for (int j = 0; j < 2; j++) {
      cin >> tr[j];
    }
    if (i == 0) {
      d += tr[1];
    } else
      d = d - tr[0] + tr[1];

    if (d > c) {
      c = d;
    }
  }

  cout << c << endl;
}