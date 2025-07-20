#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ull unsigned long long;
#define ll long long;

using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> coins(n);
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
  }
  sort(coins.begin(), coins.end());

  int lo = 0, hi = n - 1;
  int sumLo = 0, sumHi = 0;
  int loCount = 0, hiCount = 0;
  while (lo <= hi) {
    if (sumHi <= sumLo) {
      sumHi += coins[hi];
      hi--;
      hiCount++;
    } else {
      if (sumLo + coins[lo] >= sumHi) {
        sumHi += coins[hi];
        hi--;
        hiCount++;
      } else {
        sumLo += coins[lo];
        lo++;
        loCount++;
      }
    }
  }

  cout << hiCount;
  return 0;
}