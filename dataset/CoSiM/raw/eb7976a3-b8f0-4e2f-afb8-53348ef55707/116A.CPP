#include <bits/stdc++.h>

using namespace std;

int main() {
  ios::sync_with_stdio(false);

  int n;
  cin >> n;
  int out, in;
  int cur = 0;
  int max_capacity = 0;
  while (n--) {
    cin >> out >> in;
    cur = max(cur, out);
    cur -= out;
    cur += in;
    max_capacity = max(max_capacity, cur);
  }
  cout << max_capacity << endl;
}
