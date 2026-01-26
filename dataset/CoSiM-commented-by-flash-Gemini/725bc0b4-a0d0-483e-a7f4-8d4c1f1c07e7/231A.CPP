#include <bits/stdc++.h>
using namespace std;

int main() {
  int soal, yakin = 0, solusi = 0;
  cin >> soal;
  for (int i = 0; i < soal; i++) {
    int input;
    for (int i = 0; i < 3; i++) {
      cin >> input;
      if (input == 1) {
        yakin++;
      }
    }
    if (yakin >= 2) {
      solusi++;
    }
    yakin = 0;
  }
  cout << solusi;

  return 0;
}
