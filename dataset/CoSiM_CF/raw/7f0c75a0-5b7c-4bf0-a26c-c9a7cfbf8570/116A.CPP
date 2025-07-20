#include <bits/stdc++.h>
using namespace std;

int main() {
  int amt, maxn = -1, i, out, in, now = 0;
  cin >> amt;
  for (i = 1; i <= amt; i++) {
    cin >> out >> in;
    now += in - out;
    maxn = max(maxn, now);
  }
  cout << maxn << endl;
  return 0;
}
