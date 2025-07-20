// Created by Rahul Goel
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <queue>
#include <stack>
#include <string>
#include <vector>

using namespace std;
typedef long long ll;
ll mod = 1000000007;

int main() {
  ll n;
  cin >> n;
  ll c = 0, c4 = 0, c7 = 0;
  while (n) {
    if (n % 10 == 4)
      c4++;
    else if (n % 10 == 7)
      c7++;
    n /= 10;
  }
  c = c4 + c7;
  ll count = 0;
  c4 = 0, c7 = 0;
  while (c) {
    count++;
    if (c % 10 == 4)
      c4++;
    else if (c % 10 == 7)
      c7++;
    c /= 10;
  }
  if (count == c4 + c7 && count != 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}