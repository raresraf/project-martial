#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count[1000];
  bool a[1000][1000];
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      scanf("%d", &a[i][j]);
      count[i] += a[i][j];
    }
  }
  int con = 0;
  for (int i = 0; i < n; i++) {
    if (count[i] >= 2) {
      con++;
    }
  }
  cout << con << endl;
}
