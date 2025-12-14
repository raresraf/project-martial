#include <bits/stdc++.h>
using namespace std;
int main() {
  int CntI = 0, CntJ = 0;
  int data[6][6];
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> data[i][j];
      if (data[i][j] == 1) {
        if (i < 3)
          CntI = 3 - i;
        else if (i > 3)
          CntI = i - 3;
        if (j < 3)
          CntJ = 3 - j;
        else if (j > 3)
          CntJ = j - 3;
      }
    }
  }
  cout << CntI + CntJ;
  return 0;
}
