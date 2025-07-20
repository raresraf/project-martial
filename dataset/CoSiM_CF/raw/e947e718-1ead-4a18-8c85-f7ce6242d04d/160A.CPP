#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, a, sum = 0, mine = 0, coins = 0;
  vector<int> v;

  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> a;
    v.push_back(a);
  }

  for (auto x : v) {
    sum += x;
  }

  sort(v.begin(), v.end());

  while (mine <= sum) {
    mine += v.back();
    sum -= v.back();
    v.pop_back();
    coins++;
  }

  cout << coins << endl;

  return 0;
}