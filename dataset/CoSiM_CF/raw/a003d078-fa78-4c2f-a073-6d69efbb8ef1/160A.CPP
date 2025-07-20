#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<long long> v;
  long long data, sum = 0, sum1 = 0;
  int n, cnt = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> data;
    v.push_back(data);
    sum += data;
  }
  sort(v.begin(), v.end());
  for (int i = n - 1; sum1 <= sum; i--) {
    sum1 += v[i];
    sum -= v[i];
    cnt++;
    if (sum1 > sum)
      break;
  }
  cout << cnt << endl;
  return 0;
}
