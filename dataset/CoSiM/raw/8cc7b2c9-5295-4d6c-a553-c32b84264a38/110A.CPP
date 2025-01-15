#include <bits/stdc++.h>
using namespace std;
int main() {
  long long l;
  cin >> l;
  int flag = 0, m = 0;
  while (l != 0) {
    int t = l % 10;

    if (t == 4 || t == 7) {
      flag++;
    }
    l /= 10;
  }
  int k = flag;
  while (flag) {
    int x = flag % 10;
    if (x != 4 && x != 7) {
      m = 1;
      break;
    }
    flag /= 10;
  }

  if (m == 0 && k != 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
