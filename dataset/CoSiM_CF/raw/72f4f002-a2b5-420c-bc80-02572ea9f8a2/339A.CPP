#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int n, m;
  cin >> n >> m;
  std::vector<int> v(m);
  for (int i = 0; i < m; i++)
    cin >> v[i];
  long long int time = v[0] - 1;
  for (long long int i = 1; i < m; i++) {
    if (v[i] < v[i - 1]) {
      time += n - v[i - 1] + v[i];
    } else
      time += v[i] - v[i - 1];
  }
  cout << time << endl;

  return 0;
}