#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<int> v;

  int n, x;
  cin >> n;

  int you = 0, twin = 0, count = 0;
  for (; n > 0; n--) {
    cin >> x;
    v.push_back(x);
    twin += x;
  }

  sort(v.begin(), v.end(), greater<int>());

  for (int i : v) {
    you += i;
    twin -= i;
    count++;
    if (you > twin)
      break;
  }

  cout << count;

  return 0;
}
