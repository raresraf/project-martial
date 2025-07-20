// RUMAN CSE (6th Intake) BRUR
#include <algorithm>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <strstream>
#include <vector>
#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
#define PI 3.1416

using namespace std;
int main() {
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  long long k, n, w, cost, borrow;
  cin >> k >> n >> w;
  cost = k * ((w * (w + 1)) / 2);
  if (cost > n)
    borrow = cost - n;
  else
    borrow = 0;
  cout << borrow << endl;

  return 0;
}
