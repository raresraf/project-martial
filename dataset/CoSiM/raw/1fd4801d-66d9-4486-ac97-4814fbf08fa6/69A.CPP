#include <bits/stdc++.h>
using namespace std;
int cnt[10];
bool check(int n) {
  memset(cnt, 0, sizeof(cnt));
  int s = n;
  while (s > 0) {
    if (cnt[s % 10] == 1)
      return false;
    cnt[s % 10] = 1;
    s = s / 10;
  }
  return 1;
}
int main() {
  int n;
  scanf("%d", &n);
  while (1) {
    n++;
    if (check(n) == true)
      break;
  }

  cout << n << endl;
  return 0;
}
