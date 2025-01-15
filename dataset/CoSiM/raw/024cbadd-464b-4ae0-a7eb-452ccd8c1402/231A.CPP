#include <iostream>
using namespace std;
int main() {
  int n, count = 0;
  int a[1000][3];
  cin >> n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    int flag = 0;
    if (a[i][0] == 1)
      flag++;
    if (a[i][1] == 1)
      flag++;
    if (a[i][2] == 1)
      flag++;
    if (flag >= 2)
      count++;
  }
  cout << count;
  return 0;
}
