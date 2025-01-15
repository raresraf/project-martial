#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

int main() {
  int m, n;
  cin >> m >> n;
  cout << (m * n - ((m * n) % 2)) / 2;
  return 0;
}