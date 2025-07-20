#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[5][5];
  int save_i = 0, save_j = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        save_i = i + 1;
        save_j = j + 1;
      }
    }
  }
  save_i = abs(save_i - 3);
  save_j = abs(save_j - 3);

  cout << save_i + save_j << endl;
}