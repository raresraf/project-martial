#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  long long sum = 0, temp = 0;
  cin >> n;
  vector<int> v(n);
  for (int i = 0; i < n; i++) {
    cin >> v[i];
    sum += v[i];
  }
  sort(v.begin(), v.end(), greater<int>());
  int i = 0;
  while (temp <= sum / 2)
    temp += v[i++];
  cout << i << endl;
  return 0;
}
