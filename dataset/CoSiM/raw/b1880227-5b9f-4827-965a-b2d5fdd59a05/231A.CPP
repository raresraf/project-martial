#include <bits/stdc++.h>
using namespace std;
int main() {
  int N, cnt = 0, dm, res = 0;
  cin >> N;
  for (int i = 1; i <= N; i++) {
    cnt = 0;
    for (int j = 1; j <= 3; j++) {
      cin >> dm;
      if (dm == 1)
        cnt++;
    }
    if (cnt >= 2)
      res++;
  }
  cout << res << endl;
}
