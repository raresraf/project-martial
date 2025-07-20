#include <bits/stdc++.h>

using namespace std;

int main() {
  int f;
  int init;
  int bananas;

  int ret = 0;
  cin >> f >> init >> bananas;
  int eve = f;
  if (bananas == 0 || f == 0) {
    cout << 0;
    return 0;
  }
  while (bananas--) {
    ret += f;
    f += eve;
  }
  if (ret - init < 0) {
    cout << 0;
    return 0;
  }
  cout << ret - init;
  return 0;
}