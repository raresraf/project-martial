#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n;
  vector<int> v;
  cin >> n;
  long sum = 0;
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    sum += x;
    v.push_back(x);
  }
  sort(v.begin(), v.end());
  long sum2 = 0, count = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum2 += v[i];
    count++;
    if (sum2 >= sum / 2 + 1)
      break;
  }
  cout << count << "\n";
}