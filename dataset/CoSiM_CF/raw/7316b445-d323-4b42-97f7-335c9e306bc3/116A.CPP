#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int min = -1, present = 0;
  while (n--) {
    int x, y;
    cin >> x >> y;
    if ((y - x + present) > min) {
      min = (y - x + present);
    }
    present += (y - x);
    // cout<<present<<endl;
  }
  cout << min;

  return 0;
}
