#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n;
  int w, total = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    int temp = i * k;
    total += temp;
  }
  int dif = total - n;
  if (dif >= 0)
    cout << dif << endl;
  else
    cout << 0 << endl;
}
