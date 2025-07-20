#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[5][5], posi, posj, count = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        posi = i;
        posj = j;
      }
    }
  }
  /*  while(posi != 2 && posj !=2){
        if(posi > 2) { posi--; }
        if(posi < 2) { posi++; }
        if(posj > 2) { posj--; }
        if(posj < 2) { posj++; }
        count += 1;
    }
   */
  while (posj > 2) {
    posj--;
    count += 1;
  }
  while (posi < 2) {
    posi++;
    count += 1;
  }
  while (posj < 2) {
    posj++;
    count += 1;
  }
  while (posi > 2) {
    posi--;
    count += 1;
  }
  cout << count;
  return 0;
}