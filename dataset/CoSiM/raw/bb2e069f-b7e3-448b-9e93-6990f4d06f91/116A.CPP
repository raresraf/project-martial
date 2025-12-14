#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, temp = 0;
  scanf("%d", &n);
  int a, b;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    scanf("%d%d", &a, &b);
    temp = temp - a + b;
    v.push_back(temp);
  }
  sort(v.begin(), v.end());
  reverse(v.begin(), v.end());
  cout << v[0] << "\n";

  return 0;
}
