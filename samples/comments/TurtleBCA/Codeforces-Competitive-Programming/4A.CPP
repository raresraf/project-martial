#include <algorithm>
#include <bitset>
#include <climits>
#include <cmath>
#include <deque>
#include <iomanip>
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
  if (n == 2) {
    cout << "NO";
  } else {
    cout << (n % 2 == 0 ? "YES" : "NO");
  }
  return 0;
}