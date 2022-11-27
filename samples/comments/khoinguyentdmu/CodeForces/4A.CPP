// http://codeforces.com/problemset/problem/4/A

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  (n > 3 && n % 2 == 0) ? cout << "YES" : cout << "NO";
  return 0;
}
