#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  char a[150][3];
  int i, j, ans;
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++)
      cin >> a[i][j];
  }
  ans = 0;
  for (i = 0; i < n; i++) {
    if (a[i][1] == '+')
      ans++;
    else
      ans--;
  }
  cout << ans << endl;
}
