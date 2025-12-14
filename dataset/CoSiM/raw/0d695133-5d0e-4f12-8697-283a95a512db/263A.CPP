#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[5][5], i, j, i1, j1, sum;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
    }
  }
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        i1 = i;
        j1 = j;

        break;
      }
    }
  }
  // cout<<i1<<endl;
  // cout<<j1<<endl;
  if (i1 < 2) {
    i1 = 2 - i1;
  } else {
    i1 = i1 - 2;
  }
  if (j1 < 2) {
    j1 = 2 - j1;
    ;
  } else {
    j1 = j1 - 2;
  }
  sum = i1 + j1;
  cout << sum;
  return 0;
}
