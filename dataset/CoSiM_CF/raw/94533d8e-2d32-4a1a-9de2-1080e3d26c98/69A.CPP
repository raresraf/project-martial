#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  vector<int> x;
  vector<int> y;
  vector<int> z;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    x.push_back(a);
    y.push_back(b);
    z.push_back(c);
  }
  int sum_x = 0, sum_y = 0, sum_z = 0;
  for (auto i : x)
    sum_x += i;

  for (auto i : y)
    sum_y += i;

  for (auto i : z)
    sum_z += i;

  if (sum_x == 0 && sum_y == 0 && sum_z == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
}
