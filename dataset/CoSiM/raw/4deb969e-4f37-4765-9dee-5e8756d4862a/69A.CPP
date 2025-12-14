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

#define ull unsigned long long
#define ll long long

using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> sum(3);
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    sum[0] += x;
    sum[1] += y;
    sum[2] += z;
  }
  cout << (sum[0] == 0 && sum[1] == 0 && sum[2] == 0 ? "YES" : "NO");
  return 0;
}