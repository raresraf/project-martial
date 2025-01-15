#include <bits/stdc++.h>
using namespace std;
int main() {
  int posX, posY;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      int inputan;
      scanf("%d", &inputan);
      if (inputan == 1) {
        posY = i + 1;
        posX = j + 1;
      }
    }
  }
  printf("%d", abs(posX - 3) + abs(posY - 3));
}